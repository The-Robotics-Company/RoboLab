"""Pack a USD stage and everything it depends on into a self-contained folder (+ optional .usdz).

Usage: OMNI_KIT_ACCEPT_EULA=YES /opt/IsaacSim/python.sh pack_stage.py <stage.usda> <root_dir> <out_dir> [--usdz]

All dependencies (sublayers, references, payloads, textures, MDL modules) are copied into <out_dir>
preserving their paths relative to <root_dir>, so the stage's relative references keep working.
MDL modules import other MDL modules, which dependency scanning cannot see, so the whole
<root_dir>/assets/materials tree is copied as well when any MDL is referenced.
"""
import os
import shutil
import sys

sys.stdout.reconfigure(line_buffering=True)
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})
from pxr import Sdf, Usd, UsdUtils

stage_path = os.path.abspath(sys.argv[1]); root = os.path.abspath(sys.argv[2]); out = os.path.abspath(sys.argv[3])
want_usdz = "--usdz" in sys.argv
os.makedirs(out, exist_ok=True)

layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(stage_path))
files = {stage_path} | {l.realPath for l in layers if l.realPath} | {a for a in assets if a}
print(f"[pack] {len(layers)} layers, {len(assets)} assets, {len(unresolved)} unresolved")
for u in unresolved:
    print("[pack] UNRESOLVED:", u)

copied, outside = 0, []
for f in sorted(files):
    if not os.path.isfile(f):
        continue
    rel = os.path.relpath(f, root)
    if rel.startswith(".."):
        outside.append(f); continue
    dst = os.path.join(out, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(f, dst); copied += 1
print(f"[pack] copied {copied} files")
for f in outside:
    print("[pack] OUTSIDE ROOT (not copied):", f)

# MDL modules import siblings (e.g. ::Base::...), invisible to the scanner: ship the whole materials tree.
if any(f.endswith(".mdl") for f in files):
    src_mat = os.path.join(root, "assets", "materials")
    if os.path.isdir(src_mat):
        shutil.copytree(src_mat, os.path.join(out, "assets", "materials"), dirs_exist_ok=True)
        print("[pack] copied assets/materials tree for MDL imports")

if want_usdz:
    # Note: fails for RoboLab scenes because fixture USDs reference MDL modules by Omniverse cloud URL,
    # which the usdz writer cannot embed. The folder bundle is the reliable deliverable.
    usdz = os.path.join(out, os.path.splitext(os.path.basename(stage_path))[0] + ".usdz")
    try:
        ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(stage_path), usdz)
        print(f"[pack] usdz written: {usdz} ({os.path.getsize(usdz)/1e6:.1f} MB)" if ok else "[pack] usdz FAILED")
    except Exception as e:  # noqa: BLE001
        print(f"[pack] usdz FAILED, removed partial file: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        if os.path.exists(usdz):
            os.remove(usdz)

# verify: open the packed stage and count prims / check it composes without errors
packed = os.path.join(out, os.path.relpath(stage_path, root))
st = Usd.Stage.Open(packed)
n = sum(1 for _ in st.Traverse(Usd.TraverseInstanceProxies()))
_, _, unres2 = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(packed))
print(f"[pack] verify: packed stage composes {n} prims, unresolved deps: {len(unres2)}")
for u in unres2:
    print("[pack] verify UNRESOLVED:", u)
sys.stdout.flush()
_app.close()
