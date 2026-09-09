"""Headless render of a stage from named camera prims plus an overview, after settling physics.

Usage:
  OMNI_KIT_ACCEPT_EULA=YES /opt/IsaacSim/python.sh render_cams.py <stage.usda> <out_dir> \
      --cam /World/piper_x/gripper_base/wrist_camera --cam /World/exo_camera [--res 624x352] [--settle 240]
Overview views are added automatically from an extra camera framing the content bbox.
"""
import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("stage")
ap.add_argument("out_dir")
ap.add_argument("--cam", action="append", default=[], help="camera prim path (repeatable)")
ap.add_argument("--res", default="624x352")
ap.add_argument("--settle", type=int, default=240, help="physics/render frames before capture")
ap.add_argument("--no-overview", action="store_true")
ap.add_argument("--closeup", action="append", default=[], help="prim path to frame in a close-up (repeatable)")
ap.add_argument("--no-lights", action="store_true", help="do not add session-only dome/key lights")
args = ap.parse_args()
W, H = (int(v) for v in args.res.split("x"))
os.makedirs(args.out_dir, exist_ok=True)

app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": os.environ.get("RENDER_MODE", "RayTracedLighting"), "samples_per_pixel_per_frame": int(os.environ.get("RENDER_SPP", "1"))})

import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.viewports import set_camera_view

open_stage(os.path.abspath(args.stage))
stage = omni.usd.get_context().get_stage()
world = World(physics_dt=1 / 120, rendering_dt=1 / 60)
world.reset()

# Session-only lighting (never authored into the stage): headless Isaac has no default light rig.
from pxr import UsdLux
if not args.no_lights:
    dome = UsdLux.DomeLight.Define(stage, "/World/_render_dome"); dome.CreateIntensityAttr(400.0)
    key = UsdLux.DistantLight.Define(stage, "/World/_render_key"); key.CreateIntensityAttr(1500.0); key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, 0.0))

# content bbox for the overview camera (skip floor-sized prims)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
bbox = Gf.Range3d()
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
        continue
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty() or max(r.GetSize()[0], r.GetSize()[1]) > 6.0:
        continue
    bbox.UnionWith(r)
lo, hi = np.array(bbox.GetMin()), np.array(bbox.GetMax())
focus = (lo + hi) / 2; focus[2] = lo[2] + 0.75 * (hi - lo)[2]
d = float(np.linalg.norm(hi - lo)) / 2 * 1.9
print(f"[render] content bbox min={lo.round(3)} max={hi.round(3)}")

cams = list(args.cam)
overview = {}
if not args.no_overview:
    ov = UsdGeom.Camera.Define(stage, "/World/_overview_cam")
    ov.CreateFocalLengthAttr(24.0); ov.CreateHorizontalApertureAttr(20.955); ov.CreateVerticalApertureAttr(20.955 * 9 / 16)
    ov.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    overview = {
        "overview_angled": focus + np.array([-0.8 * d, -0.8 * d, 0.75 * d]),
        "overview_robot_side": focus + np.array([0.0, -1.1 * d, 0.45 * d]),
        "overview_top": focus + np.array([0.0, -0.01, 1.4 * d]),
    }
    # optional close-ups: --closeup /prim/path frames that prim's world bbox (evaluated after settling)
    closeups = [c for c in (args.closeup or [])]

# one render product + rgb annotator per camera
products = {}
for cam in cams:
    if not stage.GetPrimAtPath(cam).IsValid():
        print(f"[render] WARNING camera prim not found: {cam}")
        continue
    rp = rep.create.render_product(cam, (W, H))
    an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rp])
    products[cam] = an
if overview:
    rp = rep.create.render_product("/World/_overview_cam", (1280, 720))
    an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rp])
    products["/World/_overview_cam"] = an

print(f"[render] settling {args.settle} frames")
for _ in range(args.settle):
    world.step(render=True)

# report where the arm ended up (joint state) so the pose can be sanity-checked
try:
    from isaacsim.core.prims import Articulation
    art = Articulation("/World/piper_x")
    art.initialize()
    names = list(art.joint_names); pos = art.get_joint_positions()[0]
    print("[render] joints:", {n: round(float(p), 3) for n, p in zip(names, pos)})
except Exception as e:  # noqa: BLE001
    print(f"[render] (joint readback skipped: {type(e).__name__}: {e})")

def grab(an):
    for _ in range(4):
        world.step(render=True)
    return np.asarray(an.get_data())[:, :, :3]

name = os.path.splitext(os.path.basename(args.stage))[0]
for cam, an in products.items():
    if cam == "/World/_overview_cam":
        shots = dict(overview)
        for cp in closeups:
            prim = stage.GetPrimAtPath(cp)
            if not prim.IsValid():
                print(f"[render] WARNING closeup prim not found: {cp}"); continue
            r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(prim).ComputeAlignedRange()
            c = (np.array(r.GetMin()) + np.array(r.GetMax())) / 2; rad = float(np.linalg.norm(np.array(r.GetSize()))) / 2
            tag = "closeup_" + cp.strip("/").split("/")[-1]
            shots[tag + "_a"] = (c, c + rad * 2.6 * np.array([-0.6, -0.7, 0.4]))
            shots[tag + "_b"] = (c, c + rad * 2.6 * np.array([0.7, 0.55, 0.45]))
        for tag, eye in shots.items():
            tgt = focus
            if isinstance(eye, tuple):
                tgt, eye = eye
            set_camera_view(eye=eye, target=tgt, camera_prim_path=cam)
            for _ in range(40):
                world.step(render=True)
            rgb = grab(an)
            p = os.path.join(args.out_dir, f"{name}_{tag}.png"); Image.fromarray(rgb.astype(np.uint8)).save(p)
            print(f"[render] saved {p} shape={rgb.shape}")
    else:
        rgb = grab(an)
        tag = cam.strip("/").split("/")[-1]
        p = os.path.join(args.out_dir, f"{name}_{tag}.png"); Image.fromarray(rgb.astype(np.uint8)).save(p)
        print(f"[render] saved {p} shape={rgb.shape} mean={rgb.mean():.1f}")

print("[render] done")
world.stop()
app.close()
