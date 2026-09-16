"""Settle a stage and report, per dynamic object, how far it moved and whether it is still jittering.

A large drop means the collider sat above the visual surface (hull bulge); a large rise means penetration was pushed
out; residual speed after settling means hull seams or interpenetration. Usage:
  OMNI_KIT_ACCEPT_EULA=YES /opt/IsaacSim/python.sh stages/physics_check.py <stage.usda> [--settle 480] [--drop <obj>=<container>]
"""
import sys

sys.stdout.reconfigure(line_buffering=True)
from isaacsim import SimulationApp

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_flags = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
app = SimulationApp({"headless": True})
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import open_stage
from pxr import Usd, UsdPhysics

import os

open_stage(os.path.abspath(_args[0]))
stage = omni.usd.get_context().get_stage()
scene_root = "/World/scene" if stage.GetPrimAtPath("/World/scene").IsValid() else "/World"
dyn = []
for c in stage.GetPrimAtPath(scene_root).GetChildren():
    rb = UsdPhysics.RigidBodyAPI(c)
    kin = c.GetAttribute("physics:kinematicEnabled").Get() if c.HasAttribute("physics:kinematicEnabled") else False
    if rb and rb.GetRigidBodyEnabledAttr().Get() and not kin:
        dyn.append(c.GetName())
print(f"[phys] {os.path.basename(_args[0])}: dynamic objects {dyn}")
world = World(physics_dt=1 / 120, rendering_dt=1 / 60)
world.reset()
prims = {n: RigidPrim(f"{scene_root}/{n}") for n in dyn}
def state(rp):
    p, _ = rp.get_world_poses()
    v = rp.get_velocities()
    return np.asarray(p)[0], float(np.linalg.norm(np.asarray(v)[0][:3]))
start = {n: state(p)[0] for n, p in prims.items()}
settle = int(_flags.get("--settle", 480))
for _ in range(max(settle - 120, 0)):
    world.step(render=False)
mid = {n: state(p)[0] for n, p in prims.items()}   # 1 s before the end: drift over the last second = still moving
for _ in range(min(settle, 120)):
    world.step(render=False)
print(f"[phys] settled {settle / 120:.1f} s (drift = movement during the final 1 s; reported |v| from the sim is "
      f"unreliable when a body is at rest, so displacement is the test)")
for n, p in prims.items():
    pos, _ = state(p)
    d, late = pos - start[n], pos - mid[n]
    flag = "   <-- still moving" if np.linalg.norm(late) > 0.001 else ""
    print(f"[phys] {n:22s} dz={d[2] * 1000:+7.1f} mm  dxy={np.linalg.norm(d[:2]) * 1000:6.1f} mm  "
          f"drift={np.linalg.norm(late) * 1000:5.2f} mm{flag}")
drop = _flags.get("--drop")
if isinstance(drop, str):
    obj, cont = drop.split("=")
    cpos, _ = state(prims[cont]) if cont in prims else (np.asarray(RigidPrim(f"{scene_root}/{cont}").get_world_poses()[0])[0], 0)
    prims[obj].set_world_poses(positions=np.array([[cpos[0], cpos[1], cpos[2] + 0.2]]), orientations=np.array([[1.0, 0, 0, 0]]))
    prims[obj].set_velocities(np.zeros((1, 6)))
    for _ in range(480):
        world.step(render=False)
    opos, ospeed = state(prims[obj])
    print(f"[phys] dropped {obj} over {cont}: settled {(opos[2] - cpos[2]) * 1000:+.0f} mm above the container origin, "
          f"xy offset {np.linalg.norm(opos[:2] - cpos[:2]) * 1000:.0f} mm, |v|={ospeed * 1000:.1f} mm/s")
world.stop()
app.close()
