# SPDX-License-Identifier: Apache-2.0
"""The food_packing multi-env scene, shared by gen_demos.py and eval.py.

Import AFTER AppLauncher has started Isaac (isaaclab imports need the app).
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_ROBOTIQ_GRIPPER_CFG

from scene_physics import ALL_SCENE_OBJS, WRIST_CAM

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCENE = os.path.join(REPO, "assets", "scenes", "food_packing_opus_gt.usda")
IMG_W, IMG_H = 320, 180          # render size for BOTH demos and eval; pi05 pads to 224x224. Must match or the policy degrades.


def make_scene_cfg(wrist_cam="legacy", img_w=IMG_W, img_h=IMG_H):
    robot_cfg = FRANKA_ROBOTIQ_GRIPPER_CFG.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cam = dict(height=img_h, width=img_w, data_types=["rgb"])

    @configclass
    class FoodPackingSceneCfg(InteractiveSceneCfg):
        dome = AssetBaseCfg(prim_path="/World/Dome", spawn=sim_utils.DomeLightCfg(intensity=900.0))
        key = AssetBaseCfg(prim_path="/World/Key",
                           spawn=sim_utils.SphereLightCfg(intensity=110000.0, radius=0.06),
                           init_state=AssetBaseCfg.InitialStateCfg(pos=(0.95, 0.10, 1.25)))
        scene_objects = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/SceneObjects",
                                     spawn=sim_utils.UsdFileCfg(usd_path=SCENE))
        robot = robot_cfg                    # BEFORE the wrist camera: entities spawn in declaration order
        exo = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera", **cam,
            spawn=sim_utils.PinholeCameraCfg(focal_length=2.1, focus_distance=28.0,
                                             horizontal_aperture=5.376, vertical_aperture=3.024),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.57, 0.66),
                                            rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"))
        wrist = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/robot/Robotiq_2F_85_edit/Robotiq_2F_85/base_link/wrist_cam", **cam,
            spawn=sim_utils.PinholeCameraCfg(focal_length=2.8, focus_distance=28.0,
                                             horizontal_aperture=5.376, vertical_aperture=3.024),
            offset=TiledCameraCfg.OffsetCfg(**WRIST_CAM[wrist_cam]))

    cfg_cls = FoodPackingSceneCfg
    # EVERY object is registered so every one is reset per episode/batch. Anything left out
    # keeps the previous episode's state (measured: one robot blow-up wrecked 18 eval episodes).
    for n in ALL_SCENE_OBJS:
        setattr(cfg_cls, n, RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/SceneObjects/{n}", spawn=None))
    return cfg_cls


def build_scene(num_envs, env_spacing=14.0, wrist_cam="legacy", img_w=IMG_W, img_h=IMG_H):
    """One ROW of envs, not the default sqrt(N) grid: the exo camera looks toward +X and in a
    grid it stares at the next env's table. IsaacLab builds the cloner with spacing only, so
    force num_per_row here."""
    from isaacsim.core.cloner import GridCloner as _GC
    _init = _GC.__init__
    _GC.__init__ = lambda self, spacing, num_per_row=-1, stage=None: _init(self, spacing, num_per_row=num_envs, stage=stage)
    try:
        cfg_cls = make_scene_cfg(wrist_cam, img_w, img_h)
        scene = InteractiveScene(cfg_cls(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True))
    finally:
        _GC.__init__ = _init
    return scene
