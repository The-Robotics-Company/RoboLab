# SPDX-License-Identifier: Apache-2.0

"""AgileX PiPER-X 6-DoF arm with the coupled parallel gripper (The Robotics Company setup).

Everything control-relevant lives here, not in the USD: the asset under ``assets/robots/piper_x`` is the
untouched IsaacLab UrdfConverter output. Reference values come from the trc-spaces MuJoCo model
(``assets/piper_x/piper_x.xml``, ``asset_library/cubes_in_cup_scene.xml``) and the Orbbec DaBai DC1 calibration.

Environment switches (read at import time):
  PIPERX_WRIST_CAM = "cad" (default) | "calibrated"
      cad         -> hand-tuned mount pose from the MJCF (matches trc-spaces renders)
      calibrated  -> hand-eye result 2026-09 (15 poses, ~1 mm), incl. the principal-point offset
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg

# ----------------------------------------------------------------------------- joints / links
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = ["gripper_joint1", "gripper_joint2"]  # prismatic, joint2 = -joint1 on the real arm
# m PER FINGER. Both fingers move symmetrically (joint2 = -joint1), so the JAW GAP IS TWICE this value
# -- AgileX's own convention (piper_ros: "the range of joint7 in RViz is [0, 0.04], but the actual gripper
# range is 0.08m"). Measured on the asset: the finger pads touch at joint 0 (gap 0.0 mm) and each finger
# travels 0..0.05 m along the gripper's opening axis, so the jaw spans 0..100 mm, confirmed against the
# real arm (~10 cm fully open). NOT a 70 mm maximum: 0.035 is a chosen open command using 70 of the
# 100 mm, matching the piperx_rubiks_cube_bowl training data (gripper 0.70 x 0.05 m). Was 0.025 (cuRobo
# lock_joints target) = a 50 mm jaw, narrower than RoboLab's 58 mm rubiks_cube. Stock PiPER spec sheets
# quote a 70 mm maximum and AgileX's ROS package implies 80 mm; this arm and its URDF give 100 mm.
GRIPPER_OPEN = float(os.environ.get("PIPERX_GRIPPER_OPEN", "0.035"))
# The gripper_pos OBSERVATION normaliser is pinned separately, and by default to the value the training
# data was recorded with. Widening GRIPPER_OPEN for a grasp-tolerance experiment then leaves the proprio
# input inside its training range: fully open gives 1 - 0.048/0.035 < 0, which ObsTerm clips to 0 -- the
# same value the demonstrations record when open -- and holding the 56.8 mm cube still reads
# 1 - 0.0284/0.035 = 0.19, exactly as trained. Set PIPERX_GRIPPER_OBS_SCALE to override.
GRIPPER_OBS_SCALE = float(os.environ.get("PIPERX_GRIPPER_OBS_SCALE", "0.035"))
GRIPPER_CLOSE = 0.0    # m per finger: fully closed; the 40 N drive limit stalls on the object
GRIPPER_JOINT_COMMANDS_OPEN = {"gripper_joint1": GRIPPER_OPEN, "gripper_joint2": -GRIPPER_OPEN}
GRIPPER_JOINT_COMMANDS_CLOSE = {"gripper_joint1": GRIPPER_CLOSE, "gripper_joint2": -GRIPPER_CLOSE}
END_EFFECTOR_LINK_NAME = "gripper_base"        # flange; the gripper is a fixed child of link6
TCP_OFFSET = (0.0, 0.0, 0.12)                  # grasp_site / tcp_site in the MJCF and cuRobo ee_link
HOME_JOINT_POS = [0.0, 1.2, -1.2, 0.0, 0.0, 0.0]   # PiperXRobotConfig.init_qpos["arm"]

# ----------------------------------------------------------------------------- cameras (Orbbec DaBai DC1)
CAM_WIDTH, CAM_HEIGHT = 624, 352               # trc-spaces img_resolution for every Piper-X config
# fovy 52.5 deg (fy = 486.64 px at 480 rows) -> USD focal length 20 mm, apertures 34.9685 x 19.7258 mm
_DC1_PINHOLE = dict(focal_length=20.0, focus_distance=0.4, horizontal_aperture=34.9685, vertical_aperture=19.7258)

WRIST_CAM_SOURCE = os.environ.get("PIPERX_WRIST_CAM", "cad")
if WRIST_CAM_SOURCE == "calibrated":
    # hand-eye calibration, gripper_base -> OpenCV optical frame (= IsaacLab "ros" convention)
    _WRIST_OFFSET = TiledCameraCfg.OffsetCfg(
        pos=(-0.07223019779450245, -0.008025974405797139, 0.04782441999387742),
        rot=(0.679658, -0.17631, 0.192593, -0.685484),
        convention="ros",
    )
    # principal point cy = 214.87 of 480 rows -> 25.1 px above centre -> 5.2 % of the film gate.
    # TODO(sign): verify the direction of USD's verticalApertureOffset against a real frame before relying on it.
    _WRIST_APERTURE_OFFSET = dict(vertical_aperture_offset=-1.0327, horizontal_aperture_offset=-0.0119)
elif WRIST_CAM_SOURCE == "cad":
    # MJCF <camera name="wrist_camera" pos quat fovy="52.5"/>: MuJoCo camera frame == USD/OpenGL camera frame
    _WRIST_OFFSET = TiledCameraCfg.OffsetCfg(
        pos=(-0.07734, -0.00800, 0.04364),
        rot=(0.190415, 0.680986, -0.680986, -0.190415),
        convention="opengl",
    )
    _WRIST_APERTURE_OFFSET = {}
else:
    raise ValueError(f"PIPERX_WRIST_CAM must be 'cad' or 'calibrated', got {WRIST_CAM_SOURCE!r}")

_WRIST_CAM = TiledCameraCfg(
    # Child of gripper_base so it rides on the printed mount, exactly like the MJCF camera.
    prim_path="{ENV_REGEX_NS}/robot/gripper_base/wrist_camera",
    height=CAM_HEIGHT,
    width=CAM_WIDTH,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        clipping_range=(0.012, 100.0),   # 12 mm near clip so the finger pads are not clipped (MJCF extent=1.2 note)
        **_DC1_PINHOLE,
        **_WRIST_APERTURE_OFFSET,
    ),
    offset=_WRIST_OFFSET,
)


# ----------------------------------------------------------------------------- robot
@configclass
class PiperXCfg:
    """Fixed-base PiPER-X at the env origin (base on the table-mount plane, z = 0, facing +X)."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "piper_x", "piper_x.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,             # MJCF: gravcomp="1" on every body
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,    # MJCF handles self-contact with an exclude list; off here like the Kinova
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                **{name: value for name, value in zip(ARM_JOINT_NAMES, HOME_JOINT_POS)},
                **GRIPPER_JOINT_COMMANDS_OPEN,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        # MuJoCo position servos: gainprm=kp, biasprm=(0,-kp,-kv) -> PD with stiffness kp, damping kv.
        actuators={
            "arm_1_to_3": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3"],
                effort_limit=100.0,       # forcerange +-100 Nm
                velocity_limit=5.0,       # URDF velocity limit, rad/s
                stiffness=150.0,
                damping=15.0,
            ),
            "wrist_4_to_6": ImplicitActuatorCfg(
                joint_names_expr=["joint4", "joint5", "joint6"],
                effort_limit=30.0,        # wrist class forcerange +-30 Nm
                velocity_limit=5.0,
                stiffness=60.0,
                damping=6.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_JOINT_NAMES,
                effort_limit=40.0,        # finger class forcerange +-40 N
                velocity_limit=3.0,       # URDF, m/s
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    wrist_camera = _WRIST_CAM

    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/robot/{END_EFFECTOR_LINK_NAME}",
                name="eef_frame",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/robot/{END_EFFECTOR_LINK_NAME}",
                name="tcp_frame",
                offset=OffsetCfg(pos=TCP_OFFSET),
            ),
        ],
    )


# ----------------------------------------------------------------------------- observations
def _to_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    return wp.to_torch(value)


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    indices = [robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES]
    return _to_torch(robot.data.joint_pos)[:, indices]


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """0 = open (finger at GRIPPER_OBS_SCALE), 1 = closed, same polarity as the gripper action."""
    robot = env.scene[asset_cfg.name]
    index = robot.data.joint_names.index("gripper_joint1")
    opening = _to_torch(robot.data.joint_pos)[:, index : index + 1]
    return 1.0 - opening / GRIPPER_OBS_SCALE


def _frame_pos(env: ManagerBasedRLEnv, frames_name: str, frame: str):
    frames = env.scene[frames_name]
    index = frames.data.target_frame_names.index(frame)
    return _to_torch(frames.data.target_pos_w)[:, index, :] - env.scene.env_origins[:, :3]


def _frame_quat(env: ManagerBasedRLEnv, frames_name: str, frame: str):
    frames = env.scene[frames_name]
    index = frames.data.target_frame_names.index(frame)
    return _to_torch(frames.data.target_quat_w)[:, index, :]


def eef_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """gripper_base position in the env-local frame (= robot-root frame, base is at the origin)."""
    return _frame_pos(env, asset_cfg.name, "eef_frame")


def eef_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    return _frame_quat(env, asset_cfg.name, "eef_frame")


def tcp_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """grasp_site (gripper_base + 0.12 m along +Z) in the env-local frame."""
    return _frame_pos(env, asset_cfg.name, "tcp_frame")


def tcp_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    return _frame_quat(env, asset_cfg.name, "tcp_frame")


@configclass
class PiperXProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)                 # 6, rad
    gripper_pos = ObsTerm(func=gripper_pos, clip=(0.0, 1.0))    # 1, 0 open .. 1 closed
    eef_pos = ObsTerm(func=eef_pos)                             # 3, m, env-local
    eef_quat = ObsTerm(func=eef_quat)                           # 4, wxyz, world orientation
    tcp_pos = ObsTerm(func=tcp_pos)                             # 3, m, env-local
    tcp_quat = ObsTerm(func=tcp_quat)                           # 4, wxyz

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class PiperXWristCameraCfg:
    """Exposes the robot-mounted camera to image observation generation.

    The scene gets this sensor from ``PiperXCfg`` (so gripper_base exists before the camera is spawned);
    this wrapper only provides the observation name ``wrist_camera``. Do not pass it in ``camera_cfg``.
    """

    wrist_camera = _WRIST_CAM


# ----------------------------------------------------------------------------- actions
@configclass
class PiperXJointPositionActionCfg:
    """Absolute joint targets (rad) for joint1..6 + binary gripper: 0 opens, 1 closes."""

    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )
    gripper = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_JOINT_COMMANDS_OPEN,
        close_command_expr=GRIPPER_JOINT_COMMANDS_CLOSE,
    )


DELTA_JOINT_SCALE = 1.0  # rad per unit action; set to your training normalisation (e.g. 0.05 for +-1 -> +-0.05 rad)


@configclass
class PiperXDeltaJointPositionActionCfg:
    """Per-step joint deltas (target = current + scale * action) for joint1..6 + binary gripper."""

    arm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        scale=DELTA_JOINT_SCALE,
        use_zero_offset=True,
    )
    gripper = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_JOINT_COMMANDS_OPEN,
        close_command_expr=GRIPPER_JOINT_COMMANDS_CLOSE,
    )


# ----------------------------------------------------------------------------- contacts / friction
contact_gripper = {
    "gripper": "{ENV_REGEX_NS}/robot/gripper_link.*",
}


@configclass
class PiperXGripperFrictionEventCfg:
    """Finger-pad friction 1.0 (MJCF ``friction="1 0.05 0.002"``), applied at startup.

    The USD carries no physics material (PhysX default 0.5). IsaacLab's USD spawner cannot set friction
    from a Cfg, so this is an event term. RoboLab's factory currently takes ``events`` from the task class
    only; attach it there (``events = PiperXGripperFrictionEventCfg``) or merge it into ``env_cfg.events``
    after ``create_env`` until the factory grows an ``events_cfg`` argument.
    """

    gripper_pad_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link1", "gripper_link2"]),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


# Class-level labels, assigned after the class body so configclass does not turn them into fields.
# HDF5 EE-pose recorder channel -> body name. Default table fixture (franka_table) applies.
PiperXCfg.ee_recorder_bodies = {"ee_pose": END_EFFECTOR_LINK_NAME}
