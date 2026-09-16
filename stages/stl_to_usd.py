"""Convert a binary STL to a USD mesh asset with a simple colored material.

Usage: OMNI_KIT_ACCEPT_EULA=YES /opt/IsaacSim/python.sh stl_to_usd.py <in.stl> <out.usd> <scale> <r,g,b> [prim_name]
"""
import struct
import sys

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

src, dst, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
rgb = tuple(float(v) for v in sys.argv[4].split(","))
name = sys.argv[5] if len(sys.argv) > 5 else "mesh"

data = open(src, "rb").read()
n = struct.unpack("<I", data[80:84])[0]
rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")]))
verts = rec["v"].reshape(-1, 3).astype(np.float64) * scale
normals = np.repeat(rec["n"].astype(np.float64), 3, axis=0)
# weld identical vertices so the mesh is a proper connected surface
uniq, inverse = np.unique(verts.round(7), axis=0, return_inverse=True)
indices = inverse.reshape(-1)

stage = Usd.Stage.CreateNew(dst)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, f"/{name}")
stage.SetDefaultPrim(root.GetPrim())
mesh = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in uniq]))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * n))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices.tolist()))
mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in normals]))
mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
mesh.CreateDoubleSidedAttr(True)
ext = UsdGeom.Boundable.ComputeExtentFromPlugins(mesh, Usd.TimeCode.Default())
if ext:
    mesh.CreateExtentAttr(ext)

mat = UsdShade.Material.Define(stage, f"/{name}/Looks/{name}_mat")
sh = UsdShade.Shader.Define(stage, f"/{name}/Looks/{name}_mat/Shader")
sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
stage.GetRootLayer().Save()
print(f"[stl_to_usd] wrote {dst}: {n} tris, {len(uniq)} verts, bbox min={uniq.min(0).round(4)} max={uniq.max(0).round(4)}")
sys.stdout.flush()
_app.close()
