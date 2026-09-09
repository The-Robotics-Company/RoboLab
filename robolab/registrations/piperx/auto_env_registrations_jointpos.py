# SPDX-License-Identifier: Apache-2.0

"""Joint-position RoboLab registrations for the fixed-base PiPER-X.

Two action spaces are registered from the same robot, observation and camera setup:
  ``<Task>-PiperXJointPosition``       absolute joint targets (rad) + binary gripper
  ``<Task>-PiperXDeltaJointPosition``  per-step joint deltas + binary gripper
Observations: ``image_obs`` = {exo_camera, wrist_camera} at 624x352, ``proprio_obs`` = PiperXProprioceptionObservationCfg.
Control: physics 120 Hz, decimation 8 -> 15 Hz policy rate (same as the Kinova registration).
"""

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR

PHYSICS_DT = 1 / 120
DECIMATION = 8          # 15 Hz control
RENDER_INTERVAL = 8


_DEFAULT = object()   # sentinel: "use the Piper-X default" (None means "none", e.g. no background)

# RoboLab HDR backgrounds selectable by name; "none" keeps the untextured dome of the default lighting.
BACKGROUNDS = {
    "none": None,
    "home_office": "HomeOfficeBackgroundCfg",
    "empty_warehouse": "EmptyWarehouseBackgroundCfg",
    "billiard_hall": "BilliardHallBackgroundCfg",
    "brown_photostudio": "BrownPhotoStudioBackgroundCfg",
}


def resolve_background(name: str):
    """Map a --background name to a RoboLab background cfg class (or None)."""
    if name not in BACKGROUNDS:
        raise ValueError(f"unknown background {name!r}; choose from {sorted(BACKGROUNDS)}")
    cls = BACKGROUNDS[name]
    if cls is None:
        return None
    import robolab.variations.backgrounds as bg

    return getattr(bg, cls)


def _register(action_cfg, env_postfix, task_dirs, task, camera_cfg=None, lighting_cfg=_DEFAULT, background_cfg=_DEFAULT):
    """Default look = data-generation renders: home_office HDR dome (RoboLab HomeOfficeBackgroundCfg) + visible floor."""
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.piperx.camera_presets import PiperXExoCameraCfg
    from robolab.registrations.piperx.lighting_presets import PiperXRenderLightingCfg
    from robolab.robots.piper_x import PiperXCfg, PiperXProprioceptionObservationCfg, PiperXWristCameraCfg, contact_gripper
    from robolab.variations.camera import EgocentricMirroredWideAngleHighCameraCfg

    if lighting_cfg is _DEFAULT:
        lighting_cfg = PiperXRenderLightingCfg
    if background_cfg is _DEFAULT:
        background_cfg = resolve_background("home_office")
    camera_cfg = camera_cfg or [PiperXExoCameraCfg, EgocentricMirroredWideAngleHighCameraCfg]
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredWideAngleHighCameraCfg])
    # Policy images: exo + wrist. The wrist camera is spawned through PiperXCfg; its wrapper here only names the obs.
    ImageObsCfg = generate_image_obs_from_cameras([PiperXExoCameraCfg, PiperXWristCameraCfg])
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": PiperXProprioceptionObservationCfg(),
            "viewport_cam": ViewportCameraCfg(),
        }
    )

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_postfix=env_postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=action_cfg(),
        robot_cfg=PiperXCfg,
        camera_cfg=camera_cfg,          # never add PiperXWristCameraCfg here (would spawn before gripper_base exists)
        lighting_cfg=lighting_cfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper,
        dt=PHYSICS_DT,
        render_interval=RENDER_INTERVAL,
        decimation=DECIMATION,
        seed=1,
    )


def auto_register_piperx_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None, **kwargs):
    """Absolute joint-position envs: ``<Task>-PiperXJointPosition``."""
    from robolab.robots.piper_x import PiperXJointPositionActionCfg

    _register(PiperXJointPositionActionCfg, "PiperXJointPosition", task_dirs, task, **kwargs)


def auto_register_piperx_delta_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None, **kwargs):
    """Delta joint-position envs: ``<Task>-PiperXDeltaJointPosition``."""
    from robolab.robots.piper_x import PiperXDeltaJointPositionActionCfg

    _register(PiperXDeltaJointPositionActionCfg, "PiperXDeltaJointPosition", task_dirs, task, **kwargs)
