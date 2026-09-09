# SPDX-License-Identifier: Apache-2.0

"""Robot-agnostic lighting preset that reproduces the shared inspection stages (``stages/*.usda``).

The stages light the scene with the RoboLab HDR background alone (``robolab.variations.backgrounds``,
``HomeOfficeBackgroundCfg`` by default: dome light + backdrop, intensity 500) and make the ground visible with a
GroundPlane (UsdPreviewSurface, diffuse 0.5, default roughness 0.5) at z = -0.697, where the task tables end.
RoboLab envs get the HDR from ``background_cfg``; this preset adds only the visible ground and NO light, so an env
registered with ``lighting_cfg=StageLightingCfg`` + an HDR background renders like the stage previews on S3.

Select it in ``policies/pi0_family/run_rollout.py`` with ``--lighting stage`` (the default). ``--lighting stock``
keeps each robot's RoboLab default (DROID: SphereLightCfg 5000 at (0, -0.6, 0.7) per env + HDR background).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class StageLightingCfg:
    """Visible ground only; the light is the HDR background cfg."""

    # Spawned ONCE at /World (not per env): 100 x 100 m covers every parallel env, visual only, no collider.
    # The task scenes' own GroundPlane at z = -0.697 stays the (invisible) collider. 0.01 m slab, top face at -0.697.
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0),
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.702)),
    )
