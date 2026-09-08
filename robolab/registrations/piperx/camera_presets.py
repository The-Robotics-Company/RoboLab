# SPDX-License-Identifier: Apache-2.0

"""Scene (world-fixed) camera presets for the PiPER-X setup."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.robots.piper_x import CAM_HEIGHT, CAM_WIDTH


@configclass
class PiperXExoCameraCfg:
    """Exocentric "diag view" Orbbec DaBai DC1, world-fixed.

    Pose from trc-spaces ``asset_library/cubes_in_cup_scene.xml`` ``exo_camera``:
    pos (1.1717, -0.31, 0.5726), xyaxes (0.342 0.9397 0 | -0.3971 0.1445 0.9063), i.e. 0.878 m from the
    workspace centre (0.35, 0, 0.18) at azimuth -20.7 deg and 0.393 m above it. Intrinsics are the same DC1
    as the wrist camera (fovy 52.5 deg); the MJCF's fovy=45 does not match the real exo hardware.
    """

    exo_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/exo_camera",
        height=CAM_HEIGHT,
        width=CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=1.0,
            horizontal_aperture=34.9685,
            vertical_aperture=19.7258,
            clipping_range=(0.05, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.1717, -0.31, 0.5726),
            rot=(0.690853, 0.44014, 0.308196, 0.483751),   # MuJoCo xyaxes -> USD/OpenGL camera quaternion
            convention="opengl",
        ),
    )
