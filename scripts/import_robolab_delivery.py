# SPDX-License-Identifier: Apache-2.0

"""Convert a Robolab delivery-adapter scene into a canonical RoboLab task scene.

The delivery adapter ships objects only -- no room, floor, lighting or robot --
with every object USD referenced from a flat ``scene.usdc`` and its pose baked
into an ``xformOp:transform`` matrix. Its world origin is the capture frame, so
the support surface lands wherever the capture put it (food_packing: z = -0.312).

RoboLab task scenes instead expect, under a ``world`` default prim:
  * the work surface top at z = 0, which is where the robot's own table fixture
    (``franka_table``, top at z = 0) puts the arm's base plane;
  * ``GroundPlane`` authored at the canonical -0.697 (locked by
    tests/test_scene_ground.py);
  * a ``franka_table`` prim -- the env factory deactivates it and re-spawns the
    fixture the robot declares, so it only matters when viewing the scene alone.

This script re-authors the delivery into that form: it measures the support
surface, shifts every object up so the surface sits at z = 0, and writes the
result next to RoboLab's other scenes. Object poses are otherwise untouched, so
the delivered ground truth is preserved exactly.

Usage:
    python scripts/import_robolab_delivery.py <delivery_dir> --name food_packing
"""

import argparse
import os
import shutil

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# RoboLab scene conventions.
CANONICAL_GROUND_Z = -0.697
FRANKA_TABLE_POS = (-0.087, 0.0, 0.0)
FRANKA_TABLE_ROT = (6.123233995736766e-17, 0.0, 0.0, 1.0)  # 180 deg about +Z


def _prepend_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    """Author ``prepend apiSchemas = [...]`` on ``prim``."""
    listop = Sdf.TokenListOp()
    listop.prependedItems = schemas
    prim.SetMetadata("apiSchemas", listop)


def _author_canonical_xform(prim: Usd.Prim, matrix: Gf.Matrix4d, label: str) -> None:
    """Author ``matrix`` as translate/orient/scale ops.

    IsaacLab rejects any prim it manages whose xformOpOrder is not the canonical
    ['xformOp:translate', 'xformOp:orient', 'xformOp:scale'], so the delivery's
    single baked ``xformOp:transform`` has to be factored. The delivered matrices
    carry a Y-up -> Z-up basis change, which can make the 3x3 left-handed; that
    shows up as a negative scale component, which USD represents exactly.
    """
    xform = UsdGeom.Xform(prim)
    decomposed = Gf.Transform(matrix)

    xform.AddTranslateOp().Set(decomposed.GetTranslation())
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetRotation().GetQuat())
    xform.AddScaleOp().Set(Gf.Vec3f(*decomposed.GetScale()))

    # A factorization that does not reproduce the delivered pose would silently
    # move the object, so check it rather than trust it.
    rebuilt = xform.GetLocalTransformation(Usd.TimeCode.Default())
    error = max(abs(rebuilt[r][c] - matrix[r][c]) for r in range(4) for c in range(4))
    if error > 1e-6:
        raise ValueError(f"{label}: xform decomposition off by {error:.2e}")


def support_surface_z(prim: Usd.Prim, region: tuple[float, float, float, float]) -> float:
    """Top of the support surface inside ``region`` (xmin, xmax, ymin, ymax).

    Measured from real vertices rather than an authored extent: the delivery's
    extents are the pre-transform ones and read far too tall.
    """
    xmin, xmax, ymin, ymax = region
    best = None
    for child in Usd.PrimRange(prim):
        mesh = UsdGeom.Mesh(child)
        if not mesh:
            continue
        to_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in mesh.GetPointsAttr().Get() or []:
            world = to_world.Transform(Gf.Vec3d(point))
            if xmin < world[0] < xmax and ymin < world[1] < ymax:
                if best is None or world[2] > best:
                    best = world[2]
    if best is None:
        raise ValueError(f"no mesh vertices inside {region} for {prim.GetPath()}")
    return best


def object_bottom_z(prim: Usd.Prim) -> float:
    """Lowest vertex of ``prim`` in world space."""
    lowest = None
    for child in Usd.PrimRange(prim):
        mesh = UsdGeom.Mesh(child)
        if not mesh:
            continue
        to_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in mesh.GetPointsAttr().Get() or []:
            z = to_world.Transform(Gf.Vec3d(point))[2]
            if lowest is None or z < lowest:
                lowest = z
    return lowest


def build(delivery_dir: str, out_scene: str, asset_subdir: str, names: dict[str, str], support: str,
          region: tuple[float, float, float, float]) -> None:
    src = Usd.Stage.Open(os.path.join(delivery_dir, "scene.usdc"))
    src_root = src.GetDefaultPrim()

    support_prim = src_root.GetChild(support)
    dz = -support_surface_z(support_prim, region)
    print(f"[import] support surface z={-dz:+.4f} -> shifting scene by {dz:+.4f} m")

    stage = Usd.Stage.CreateNew(out_scene)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world = UsdGeom.Xform.Define(stage, "/world")
    stage.SetDefaultPrim(world.GetPrim())

    physics_material = stage.DefinePrim("/world/PhysicsMaterial", "Material")
    _prepend_api_schemas(physics_material, ["PhysicsMaterialAPI", "PhysxMaterialAPI"])
    physics_material.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(2.0)
    physics_material.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(2.0)
    physics_material.CreateAttribute(
        "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform
    ).Set("max")

    for src_name, dst_name in names.items():
        src_prim = src_root.GetChild(src_name)
        if not src_prim:
            raise ValueError(f"{src_name} missing from delivery")

        prim = stage.DefinePrim(f"/world/{dst_name}", "Xform")
        prim.GetPayloads().AddPayload(f"../objects/{asset_subdir}/{src_name}.usd")
        UsdPhysics.RigidBodyAPI.Apply(prim)

        matrix = Gf.Matrix4d(src_prim.GetAttribute("xformOp:transform").Get())
        translation = matrix.ExtractTranslation()
        matrix.SetTranslateOnly(Gf.Vec3d(translation[0], translation[1], translation[2] + dz))
        _author_canonical_xform(prim, matrix, dst_name)

        if src_prim.GetAttribute("physics:kinematicEnabled").Get():
            prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

        bottom = object_bottom_z(src_prim) + dz
        print(f"[import] {dst_name:<16} bottom z = {bottom:+.4f}")

    fixture = stage.DefinePrim("/world/franka_table", "Xform")
    fixture.GetPayloads().AddPayload("../fixtures/franka_table.usd")
    # The payload root already declares xformOpOrder [translate, orient, scale] at
    # identity, so override those op values rather than adding ops of our own
    # (AddXformOp would collide with the composed order). Same as colored_blocks.usda.
    fixture.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*FRANKA_TABLE_POS))
    fixture.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd).Set(
        Gf.Quatd(FRANKA_TABLE_ROT[0], Gf.Vec3d(*FRANKA_TABLE_ROT[1:]))
    )

    ground = UsdGeom.Xform.Define(stage, "/world/GroundPlane")
    ground.CreateVisibilityAttr("invisible")
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CANONICAL_GROUND_Z))
    ground.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    ground.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    mesh = UsdGeom.Mesh.Define(stage, "/world/GroundPlane/CollisionMesh")
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
    mesh.CreatePointsAttr([(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)])
    mesh.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])
    plane = stage.DefinePrim("/world/GroundPlane/CollisionPlane", "Plane")
    _prepend_api_schemas(plane, ["PhysicsCollisionAPI"])
    plane.CreateAttribute("axis", Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform).Set("Z")
    plane.CreateAttribute("purpose", Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform).Set("guide")

    stage.GetRootLayer().Save()
    print(f"[import] wrote {out_scene}")


FOOD_PACKING = {
    "obj_000000": "grey_bin_right",   # y = -0.41, i.e. the robot's right
    "obj_000001": "cheezit_box",
    "obj_000002": "soup_can",
    "obj_000003": "spam_can",
    "obj_000004": "sugar_box",
    "obj_000005": "container",
    "obj_000006": "mustard_bottle",
    "obj_000007": "grey_bin_left",    # y = +0.39
    "obj_000008": "table",
    "obj_000009": "small_box",
}


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery_dir", help="directory holding scene.usdc + meshes/")
    parser.add_argument("--name", default="food_packing")
    parser.add_argument("--copy-meshes", action="store_true", help="copy meshes/ into assets/objects/<name>/")
    args = parser.parse_args()

    asset_dir = os.path.join(here, "assets", "objects", args.name)
    if args.copy_meshes:
        os.makedirs(asset_dir, exist_ok=True)
        for f in os.listdir(os.path.join(args.delivery_dir, "meshes")):
            shutil.copy2(os.path.join(args.delivery_dir, "meshes", f), asset_dir)
        print(f"[import] copied meshes -> {asset_dir}")

    out_scene = os.path.join(here, "assets", "scenes", f"{args.name}.usda")
    if os.path.exists(out_scene):
        os.remove(out_scene)

    build(
        delivery_dir=args.delivery_dir,
        out_scene=out_scene,
        asset_subdir=args.name,
        names=FOOD_PACKING,
        support="obj_000008",
        region=(0.30, 0.75, -0.25, 0.35),
    )


if __name__ == "__main__":
    main()
