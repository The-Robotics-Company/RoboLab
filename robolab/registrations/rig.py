# SPDX-License-Identifier: Apache-2.0

"""Scene rigs: the look of a RoboLab env (light + backdrop + visible ground) loaded from ONE USD file.

A rig USD (``stages/rigs/<name>.usda``) holds every prim that is about appearance rather than task objects: the HDR
dome light (which is also the backdrop), any other lights, and the visual-only ground plane. The same file is
referenced by the inspection stages (``stages/*.usda``) and, through :func:`rig_cfg`, spawned once at ``/World/rig``
in every registered env, so preview renders and rollouts are lit by identical prims.

Why a separate file and not the scene USD: RoboLab does not reference scene USDs, it scrapes them into per-object
asset cfgs (lights are dropped), and every env gets its own copy of the scene, so a dome light inside the scene
would be cloned N times and stack N x the exposure. Global things must be spawned once at /World; that is this cfg.

Usage in a registration: ``lighting_cfg=rig_cfg("home_office"), background_cfg=None`` (the rig already carries the HDR).
``rig_cfg(None)`` / ``resolve_rig("stock")`` mean "no rig": the registration falls back to RoboLab's own
lighting_cfg / background_cfg defaults.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from robolab.constants import ASSET_DIR

RIG_DIR = os.path.normpath(os.path.join(ASSET_DIR, os.pardir, "stages", "rigs"))
DEFAULT_RIG = "home_office"


def resolve_rig(name: str | None) -> str | None:
    """Map ``--rig`` to a USD path: a rig name (``home_office`` -> stages/rigs/home_office.usda), an explicit .usd/.usda
    path, or ``stock`` / ``none`` / None for no rig (RoboLab's per-robot default lighting + background)."""
    if name is None or name.lower() in ("stock", "none"):
        return None
    if name.endswith((".usd", ".usda", ".usdc")):
        path = os.path.abspath(os.path.expanduser(name))
    else:
        path = os.path.join(RIG_DIR, f"{name}.usda")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"rig USD not found: {path} (known rigs: {sorted(known_rigs())})")
    return path


def known_rigs() -> list[str]:
    if not os.path.isdir(RIG_DIR):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(RIG_DIR) if f.endswith((".usd", ".usda", ".usdc"))]


def rig_cfg(name_or_path: str | None = DEFAULT_RIG):
    """Return a configclass spawning the rig USD once at /World/rig (usable as ``lighting_cfg``), or None for no rig."""
    usd_path = resolve_rig(name_or_path)
    if usd_path is None:
        return None
    rig_name = os.path.splitext(os.path.basename(usd_path))[0]

    @configclass
    class RigCfg:
        rig = AssetBaseCfg(
            prim_path="/World/rig",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        )

    RigCfg.__name__ = RigCfg.__qualname__ = f"Rig_{rig_name}_Cfg"
    RigCfg.rig_usd_path = usd_path
    return RigCfg
