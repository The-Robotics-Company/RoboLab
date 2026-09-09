"""Render a stage from a list of look-at camera views with given intrinsics (headless Isaac Sim).

Usage:
  OMNI_KIT_ACCEPT_EULA=YES /opt/IsaacSim/python.sh render_views.py <stage.usda> <out_dir> <views.json> \
      [--res 640x480] [--focal 24.0] [--haperture 20.955] [--settle 240] [--no-lights]
views.json: [{"name": "angled", "position": [x,y,z], "target": [x,y,z]}, ...]
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("stage"); ap.add_argument("out_dir"); ap.add_argument("views")
ap.add_argument("--res", default="640x480"); ap.add_argument("--focal", type=float, default=24.0)
ap.add_argument("--haperture", type=float, default=20.955); ap.add_argument("--settle", type=int, default=240)
ap.add_argument("--no-lights", action="store_true")
args = ap.parse_args()
W, H = (int(v) for v in args.res.split("x"))
os.makedirs(args.out_dir, exist_ok=True)
views = json.load(open(args.views))

app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RayTracedLighting"})
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.viewports import set_camera_view

open_stage(os.path.abspath(args.stage))
stage = omni.usd.get_context().get_stage()
world = World(physics_dt=1 / 120, rendering_dt=1 / 60); world.reset()
if not args.no_lights:
    UsdLux.DomeLight.Define(stage, "/World/_render_dome").CreateIntensityAttr(400.0)

cam = UsdGeom.Camera.Define(stage, "/World/_view_cam")
cam.CreateFocalLengthAttr(args.focal); cam.CreateHorizontalApertureAttr(args.haperture)
cam.CreateVerticalApertureAttr(args.haperture * H / W); cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
rp = rep.create.render_product("/World/_view_cam", (W, H))
an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rp])

for _ in range(args.settle):
    world.step(render=True)
name = os.path.splitext(os.path.basename(args.stage))[0]
for v in views:
    set_camera_view(eye=np.array(v["position"], float), target=np.array(v["target"], float), camera_prim_path="/World/_view_cam")
    for _ in range(40):
        world.step(render=True)
    rgb = np.asarray(an.get_data())[:, :, :3]
    p = os.path.join(args.out_dir, f"{name}_{v['name']}.png"); Image.fromarray(rgb.astype(np.uint8)).save(p)
    print(f"[render] saved {p} mean={rgb.mean():.1f}")
print("[render] done"); world.stop(); app.close()
