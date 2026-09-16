# SPDX-License-Identifier: Apache-2.0

"""Scene rigs: the look of a RoboLab env (light + backdrop + visible ground) loaded from ONE USD file.

A rig USD (``stages/rigs/<name>.usda``) holds every prim that is about appearance rather than task objects: the HDR
dome light (which is also the backdrop), any other lights, and the visual-only ground plane, authored at the rig's own
origin. The rig prim is then placed at the scene's ground height, so one rig file serves every scene. The same file is
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
DEFAULT_GROUND_Z = -0.697   # rubiks_cube_bowl and most RoboLab table scenes


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


def scene_ground_z(scene_usd: str, default: float = DEFAULT_GROUND_Z) -> float:
    """The z of a task scene's authored ``/GroundPlane`` (its invisible collider ground), else ``default``."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(scene_usd)
    if stage is None:
        return default
    root = stage.GetDefaultPrim()
    ground = stage.GetPrimAtPath(root.GetPath().AppendChild("GroundPlane")) if root.IsValid() else None
    if ground is None or not ground.IsValid():
        return default
    m = UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return float(m.ExtractTranslation()[2])


def rig_cfg(name_or_path: str | None = DEFAULT_RIG, ground_z: float = DEFAULT_GROUND_Z):
    """Return a configclass spawning the rig USD once at /World/rig (usable as ``lighting_cfg``), or None for no rig.

    ``ground_z`` places the rig prim, i.e. its visible ground plane, at the task scene's ground height; the dome is
    infinite so it is unaffected. Use :func:`scene_ground_z` to read that height off the scene USD.
    """
    usd_path = resolve_rig(name_or_path)
    if usd_path is None:
        return None
    rig_name = os.path.splitext(os.path.basename(usd_path))[0]

    @configclass
    class RigCfg:
        rig = AssetBaseCfg(
            prim_path="/World/rig",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
        )

    RigCfg.__name__ = RigCfg.__qualname__ = f"Rig_{rig_name}_Cfg"
    # Do NOT hang extra attributes on the class: configclass copies class attributes onto the scene cfg instance and
    # InteractiveScene rejects members that are not asset cfgs ("Unknown asset config type for ...").
    _RIG_PATHS[RigCfg] = usd_path
    _RIG_GROUND[RigCfg] = ground_z
    return RigCfg


_RIG_PATHS: dict[type, str] = {}
_RIG_GROUND: dict[type, float] = {}


def rig_usd_path(cfg_cls) -> str | None:
    """The USD file a cfg returned by :func:`rig_cfg` spawns."""
    return _RIG_PATHS.get(cfg_cls)


def rig_ground_z(cfg_cls) -> float | None:
    """The ground height a cfg returned by :func:`rig_cfg` places the rig at."""
    return _RIG_GROUND.get(cfg_cls)
