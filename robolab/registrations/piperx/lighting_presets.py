# SPDX-License-Identifier: Apache-2.0

"""Lighting presets for the PiPER-X registrations.

The Piper-X data-generation look (colleague's render pipeline, RoboLab ``render_utils`` with a background texture)
is: an HDR dome as the only light and as the background, plus the scene's floor made visible. In RoboLab envs the
HDR dome comes from ``robolab.variations.backgrounds`` (``HomeOfficeBackgroundCfg`` etc., spawned once at
``/World/background``), which ``run_rollout.py --background`` selects (default ``home_office``). This preset only
adds what RoboLab envs lack: a visible floor. Task scenes carry an invisible collision-only GroundPlane at
z = -0.697, so a matte grey visual-only slab is spawned once at /World with its top face at that height.
No lights are added here: with an HDR background the dome is the light, as in the data-generation renders.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class PiperXRenderLightingCfg:
    """Visible floor only (lighting comes from the HDR background cfg)."""

    # Matches the look of the scene-authored GroundPlane mesh (plain grey, no material) that the data-generation
    # renders force visible. 100 x 100 m so it covers every parallel env; visual only, no collider.
    floor = AssetBaseCfg(
        prim_path="/World/floor",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.9, metallic=0.0),
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.702)),
    )
