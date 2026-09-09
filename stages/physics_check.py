"""Physics check of the v0 scene colliders: settle, then drop the cube into the bowl."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.prims import RigidPrim
import omni.usd
from pxr import UsdGeom, Usd, Gf
import os
ok = open_stage(os.path.abspath(sys.argv[1]))
stage = omni.usd.get_context().get_stage()
print("[phys] open_stage ->", ok, "root children:", [c.GetPath().pathString for c in stage.GetPseudoRoot().GetChildren()], "World children:", [c.GetName() for c in stage.GetPrimAtPath("/World").GetChildren()] if stage.GetPrimAtPath("/World") else None)
world = World(physics_dt=1 / 120, rendering_dt=1 / 60); world.reset()
sc = "/World/scene" if stage.GetPrimAtPath("/World/scene").IsValid() else "/World"
print("[phys] scene children:", [c.GetName() for c in stage.GetPrimAtPath(sc).GetChildren()])
bowl = RigidPrim(f"{sc}/bowl"); cube = RigidPrim(f"{sc}/rubiks_cube")
def pos(rp): p, _ = rp.get_world_poses(); return np.asarray(p)[0]
print(f"[phys] t=0    bowl z={pos(bowl)[2]:.4f} cube z={pos(cube)[2]:.4f} (origins)")
for i in range(360):
    world.step(render=False)
b0, c0 = pos(bowl), pos(cube)
print(f"[phys] settled 3 s: bowl z={b0[2]:.4f} cube z={c0[2]:.4f}  (drop = still resting on the table? no fall-through if z > -0.1)")
# drop the cube above the bowl centre
cube.set_world_poses(positions=np.array([[b0[0], b0[1], b0[2] + 0.20]]), orientations=np.array([[1, 0, 0, 0]]))
cube.set_velocities(np.zeros((1, 6)))
for i in range(480):
    world.step(render=False)
b1, c1 = pos(bowl), pos(cube)
print(f"[phys] cube dropped into bowl, 4 s later: cube z={c1[2]:.4f} bowl z={b1[2]:.4f}; cube centre {1000*(c1[2]-b1[2]):.0f} mm above bowl origin, "
      f"xy offset {1000*np.linalg.norm(c1[:2]-b1[:2]):.0f} mm")
print("[phys] bowl rim top z ~0.100, bowl bottom z ~0.007, cube half-height ~0.03: inside the cavity means cube z well below the rim")
world.stop(); app.close()
