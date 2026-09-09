"""Bare v0 scene from the video reconstruction: essential assets only (table, bowl, rubiks_cube + collider ground).
Room, lights, PhysicsScene and room materials are dropped; look comes from a rig USD at rollout/inspection time.
Object xforms are re-authored as translate/orient/scale (RoboLab rejects a bare xformOp:transform matrix)."""
import os, shutil, sys
sys.stdout.reconfigure(line_buffering=True)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

SRC_DIR = os.path.expanduser("~/Downloads/rubiks_cube_bowl_video_scene_postadjusted_gtframe")
DST_DIR = os.path.abspath("assets/scenes/rubiks_cube_bowl_v0")
DST = os.path.join(DST_DIR, "rubiks_cube_bowl_v0.usda")
RENAME = {"obj_000000": "table", "obj_000001": "bowl", "obj_000002": "rubiks_cube"}
FLOOR_Z = -0.945   # the video room's floor height (Room/Floor mesh), used for the collider ground

os.makedirs(DST_DIR, exist_ok=True)
shutil.copytree(os.path.join(SRC_DIR, "meshes"), os.path.join(DST_DIR, "meshes"), dirs_exist_ok=True)

src = Usd.Stage.Open(os.path.join(SRC_DIR, "scene.usdc"))
src_layer = src.GetRootLayer()
src_root = src.GetDefaultPrim().GetPath()

dst = Usd.Stage.CreateNew(DST)
dst_layer = dst.GetRootLayer()
UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(dst, 1.0)
world = UsdGeom.Xform.Define(dst, "/World"); dst.SetDefaultPrim(world.GetPrim())
dst_layer.documentation = ("rubiks_cube_bowl_v0: bare video-reconstructed scene (postadjusted, GT frame) for RubiksCubeTask. "
    "Essential assets only: table (kinematic), bowl + rubiks_cube (dynamic), invisible collider ground at the video floor height. "
    "Room, lights and PhysicsScene removed; light/backdrop/visible ground come from a rig USD (stages/rigs/). "
    f"Source: {os.path.basename(SRC_DIR)}/scene.usdc")

for old, new in RENAME.items():
    Sdf.CopySpec(src_layer, src_root.AppendChild(old), dst_layer, Sdf.Path("/World").AppendChild(new))
    prim = dst.GetPrimAtPath(f"/World/{new}")
    # retarget the mesh reference (was meshes/obj_X.usd relative to scene.usdc; same layout next to the new file)
    refs = prim.GetReferences(); refs.ClearReferences()
    refs.AddReference(f"./meshes/{old}.usd")
    prim.CreateAttribute("description", Sdf.ValueTypeNames.String).Set(new.replace("_", " "))
    # RoboLab requires standard translate / orient / scale ops; the reconstruction authors one xformOp:transform matrix.
    x = UsdGeom.Xformable(prim); M = x.GetLocalTransformation(); t = Gf.Transform(M)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(t.GetTranslation()))
    q = t.GetRotation().GetQuat(); x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q.GetReal(), q.GetImaginary()))
    x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(t.GetScale()))
    if prim.HasProperty("xformOp:transform"):
        prim.RemoveProperty("xformOp:transform")
    print(f"[v0] {old} -> /World/{new}  (xform standardised: scale {[round(v, 4) for v in t.GetScale()]})")

# RoboLab-style ground: invisible collider only (visual ground lives in the rig)
gp = UsdGeom.Xform.Define(dst, "/World/GroundPlane")
gp.GetPrim().GetAttribute("visibility").Set("invisible") if gp.GetPrim().HasAttribute("visibility") else UsdGeom.Imageable(gp).CreateVisibilityAttr("invisible")
gp.AddTranslateOp().Set(Gf.Vec3d(0, 0, FLOOR_Z))
mesh = UsdGeom.Mesh.Define(dst, "/World/GroundPlane/CollisionMesh")
mesh.CreatePointsAttr([(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)])
mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
mesh.CreateNormalsAttr([(0, 0, 1)] * 4); mesh.CreateDoubleSidedAttr(False)
plane = UsdGeom.Plane.Define(dst, "/World/GroundPlane/CollisionPlane")
plane.CreateAxisAttr("Z"); plane.CreatePurposeAttr("guide")
UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
dst.GetRootLayer().Save()
print("[v0] wrote", DST)

# verify: reopen, list children of the default prim with physics flags + world bbox
chk = Usd.Stage.Open(DST); cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for c in chk.GetDefaultPrim().GetChildren():
    rb = UsdPhysics.RigidBodyAPI(c); kin = c.GetAttribute("physics:kinematicEnabled").Get() if c.HasAttribute("physics:kinematicEnabled") else None
    r = cache.ComputeWorldBound(c).ComputeAlignedRange()
    print(f"[v0] {c.GetPath()} rigid={bool(rb and rb.GetRigidBodyEnabledAttr().Get())} kinematic={kin} "
          f"bbox min={[round(v,3) for v in r.GetMin()]} max={[round(v,3) for v in r.GetMax()]}")
from pxr import UsdUtils
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(DST))
print("[v0] deps:", len(layers), "layers", len(assets), "assets, unresolved:", unresolved)
app.close()
