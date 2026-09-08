# SPDX-License-Identifier: Apache-2.0

"""Lighting presets for the PiPER-X registrations.

``PiperXRenderLightingCfg`` reproduces the lights RoboLab's own render utility adds when it renders scene
previews (``robolab/core/utils/render_utils.py::render_stage_frame`` with ``add_lighting=True`` and no
background texture): a 7250 K distant light at exposure 10 and an untextured 6150 K dome at exposure 9.
The untextured dome is also what the cameras see behind the scene, i.e. a plain light-grey ground instead
of an HDR room. Used as the default so Piper-X rollouts match the colleague's data-generation renders.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class PiperXRenderLightingCfg:
    """render_utils.render_stage_frame lighting: distant key + untextured dome, no background texture."""

    # Distant and dome lights illuminate the whole stage regardless of where their prim lives, so they must be
    # spawned ONCE at /World, not per env: under {ENV_REGEX_NS} every parallel env adds another copy and the
    # exposure scales with --num-envs (5 envs rendered ~5x too bright, floor and dome blown to white).
    distant_light = AssetBaseCfg(
        prim_path="/World/distant_light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            enable_color_temperature=True,
            color_temperature=7250.0,
            intensity=1.0,
            exposure=10.0,
            angle=30.0,
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            enable_color_temperature=True,
            color_temperature=6150.0,
            intensity=1.0,
            exposure=9.0,
            texture_file=None,
            texture_format="latlong",
        ),
    )
    # Ground: equivalent of render_utils add_ground=True = isaacsim GroundPlane: 100 x 100 m plane with a
    # UsdPreviewSurface of diffuseColor 0.5 and default roughness 0.5 / metallic 0 (the glossy, reflective look).
    # RoboLab task scenes carry only an invisible collision ground at z = -0.697, so this is a visual-only slab
    # whose top face sits exactly there; the scene's own ground plane keeps doing the physics. One slab at /World
    # covers every env (env origins share z = 0; only x/y differ).
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0),
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.702)),
    )
