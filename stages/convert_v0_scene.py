"""Bare v0 RoboLab scene from a video reconstruction: essential assets only.

Drops the room, lights, PhysicsScene and room materials (light/backdrop/visible ground come from a rig USD at
rollout/inspection time), renames the obj_00000N prims to the names the task expects, authors the franka_table robot
mount and an invisible collider GroundPlane at the room floor height, and re-authors each object's xform as standard
translate/orient/scale ops (RoboLab rejects a bare xformOp:transform matrix).

Usage:
  OMNI_KIT_ACCEPT_EULA=YES /opt/IsaacSim/python.sh stages/convert_v0_scene.py <src_dir> <out_name> \
      obj_000000=table obj_000001=bowl ... [--floor-z auto|<z>]
The mapping must be given explicitly: the delivered USD does not label its objects (identify them from
scene_info.txt masses/static flags, the albedo textures and, when the scene is in the GT frame, object positions).
"""
import os
import shutil
import sys

sys.stdout.reconfigure(line_buffering=True)
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a.split("=")[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
SRC_DIR = os.path.abspath(os.path.expanduser(args[0]))
OUT_NAME = args[1]
RENAME = dict(kv.split("=") for kv in args[2:])
DST_DIR = os.path.abspath(os.path.join("assets", "scenes", OUT_NAME))
DST = os.path.join(DST_DIR, f"{OUT_NAME}.usda")

os.makedirs(DST_DIR, exist_ok=True)
shutil.copytree(os.path.join(SRC_DIR, "meshes"), os.path.join(DST_DIR, "meshes"), dirs_exist_ok=True)

src = Usd.Stage.Open(os.path.join(SRC_DIR, "scene.usdc"))
src_layer, src_root = src.GetRootLayer(), src.GetDefaultPrim().GetPath()

# floor height: the reconstruction's Room/Floor mesh, unless given
floor_z = flags.get("--floor-z", "auto")
if floor_z == "auto":
    fl = src.GetPrimAtPath(src_root.AppendChild("Room").AppendChild("Floor"))
    if not fl.IsValid():
        raise SystemExit("no Room/Floor in the source scene; pass --floor-z=<z>")
    m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(fl)
    floor_z = round(float(min(m.Transform(Gf.Vec3d(v))[2] for v in UsdGeom.Mesh(fl).GetPointsAttr().Get())), 4)
else:
    floor_z = float(floor_z)

dst = Usd.Stage.CreateNew(DST) if not os.path.exists(DST) else Usd.Stage.Open(DST)
dst_layer = dst.GetRootLayer()
dst_layer.Clear()
UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(dst, 1.0)
world = UsdGeom.Xform.Define(dst, "/World")
dst.SetDefaultPrim(world.GetPrim())
dst_layer.documentation = (
    f"{OUT_NAME}: bare video-reconstructed scene (postadjusted, GT frame). Essential assets only: "
    f"{', '.join(RENAME.values())}; franka_table robot mount authored like every RoboLab task scene (the factory strips "
    f"and re-adds it); invisible collider GroundPlane at the room floor z = {floor_z}. Room, lights, PhysicsScene and room "
    f"materials removed; light/backdrop/visible ground come from a rig USD (stages/rigs/). Object xforms re-authored as "
    f"translate/orient/scale. Source: {os.path.basename(SRC_DIR)}/scene.usdc"
)

for old, new in RENAME.items():
    Sdf.CopySpec(src_layer, src_root.AppendChild(old), dst_layer, Sdf.Path("/World").AppendChild(new))
    prim = dst.GetPrimAtPath(f"/World/{new}")
    refs = prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(f"./meshes/{old}.usd")
    prim.CreateAttribute("description", Sdf.ValueTypeNames.String).Set(new.replace("_", " "))
    x = UsdGeom.Xformable(prim)
    t = Gf.Transform(x.GetLocalTransformation())
    q = t.GetRotation().GetQuat()
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(t.GetTranslation()))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q.GetReal(), q.GetImaginary()))
    x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(t.GetScale()))
    if prim.HasProperty("xformOp:transform"):
        prim.RemoveProperty("xformOp:transform")
    print(f"[v0] {old} -> /World/{new} (scale {[round(v, 4) for v in t.GetScale()]})")

# robot mount, exactly as the RoboLab task scenes author it
ft = dst.DefinePrim("/World/franka_table")
ft.GetPayloads().AddPayload("../../fixtures/franka_table.usd")
# Set the op values only: the payload already carries xformOpOrder (adding ops would duplicate them). Same as the
# GT task scenes, which author just xformOp:translate + xformOp:orient under the payload.
ft.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(-0.087, 0, 0))
ft.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd).Set(Gf.Quatd(6.123233995736766e-17, 0, 0, 1))

# RoboLab-style ground: invisible collider only (the visible ground is the rig's)
gp = UsdGeom.Xform.Define(dst, "/World/GroundPlane")
UsdGeom.Imageable(gp).CreateVisibilityAttr("invisible")
gp.AddTranslateOp().Set(Gf.Vec3d(0, 0, floor_z))
mesh = UsdGeom.Mesh.Define(dst, "/World/GroundPlane/CollisionMesh")
mesh.CreatePointsAttr([(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)])
mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
mesh.CreateNormalsAttr([(0, 0, 1)] * 4); mesh.CreateDoubleSidedAttr(False)
plane = UsdGeom.Plane.Define(dst, "/World/GroundPlane/CollisionPlane")
plane.CreateAxisAttr("Z"); plane.CreatePurposeAttr("guide")
UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

# faithful-to-mesh colliders: keep convex decomposition, tuned so the hulls track the reconstructed surfaces
from pxr import PhysxSchema
for new in RENAME.values():
    m = dst.OverridePrim(f"/World/{new}/Mesh")
    UsdPhysics.MeshCollisionAPI(m).CreateApproximationAttr("convexDecomposition")
    api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(m)
    api.CreateMaxConvexHullsAttr(128); api.CreateVoxelResolutionAttr(4000000)
    api.CreateErrorPercentageAttr(0.5); api.CreateHullVertexLimitAttr(64); api.CreateShrinkWrapAttr(True)
dst_layer.Save()
print(f"[v0] wrote {DST} (floor z={floor_z})")

# author extents on the copied mesh files (the delivered ones have none)
import glob
for f in sorted(glob.glob(os.path.join(DST_DIR, "meshes", "obj_*.usd"))):
    ms = Usd.Stage.Open(f)
    for p in ms.Traverse():
        if p.IsA(UsdGeom.Mesh):
            mm = UsdGeom.Mesh(p)
            if not mm.GetExtentAttr().HasAuthoredValue():
                from pxr import Vt
                ext = UsdGeom.Boundable.ComputeExtentFromPlugins(mm, Usd.TimeCode.Default())
                mm.GetExtentAttr().Set(Vt.Vec3fArray([ext[0], ext[1]]))
    ms.GetRootLayer().Save()

chk = Usd.Stage.Open(DST)
xf = UsdGeom.XformCache(Usd.TimeCode.Default())
for c in chk.GetDefaultPrim().GetChildren():
    rb = UsdPhysics.RigidBodyAPI(c)
    kin = c.GetAttribute("physics:kinematicEnabled").Get() if c.HasAttribute("physics:kinematicEnabled") else None
    P = []
    for q in Usd.PrimRange(c):
        if q.IsA(UsdGeom.Mesh):
            m = xf.GetLocalToWorldTransform(q)
            P += [[*m.Transform(Gf.Vec3d(v))] for v in UsdGeom.Mesh(q).GetPointsAttr().Get()]
    ext = (np.array(P).max(0) - np.array(P).min(0)).round(3) if P else None
    print(f"[v0] check {c.GetName():22s} rigid={bool(rb and rb.GetRigidBodyEnabledAttr().Get())} kin={kin} size={ext}")
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(DST))
print(f"[v0] deps: {len(layers)} layers, {len(assets)} assets, unresolved: {unresolved}")
app.close()
