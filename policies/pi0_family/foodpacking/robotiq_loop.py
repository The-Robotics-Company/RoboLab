"""Close the Robotiq 2F-85 four-bar loop that NVIDIA's USD leaves open.

The Isaac USD models the gripper as a *tree*:
    base_link -> outer_knuckle -> outer_finger -> inner_finger -> inner_knuckle
so each ``inner_knuckle``'s palm-side end has NO joint back to ``base_link`` -- it is a
free-ended leaf that visibly "falls off" the body, and the PhysX mimic joints (which only
make the angles track ``finger_joint``) don't pin it. PhysX reduced-coordinate
articulations can't contain a loop, so we add the missing hinge as a *maximal-coordinate*
point-to-point constraint (D6 with translations locked, rotations free) between each
inner_knuckle and base_link at the real palm pivot. The mimic still drives the angles; the
constraint just keeps the loop physically closed.

Effect (verified on the probe): the gripper goes from a distorted 8->70 mm stroke to the
true 0->85 mm 2F-85 range, and the inner links stay attached.

Call AFTER the robot prim is spawned (e.g. after InteractiveScene construction) and BEFORE
``sim.reset()``. pivot_y/_z are the base->inner_knuckle hinge offset from the Robotiq URDF.
"""

from __future__ import annotations


def author_robotiq_loop_closure(pivot_y: float = 0.0127, pivot_z: float = 0.06142,
                                x_axis_forward: bool = True) -> int:
    """Add inner_knuckle<->base_link loop-closure joints to every Robotiq 2F-85 on the stage.

    Returns the number of loop joints added.
    """
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def find_all(name):
        out = []
        for p in stage.Traverse():
            sp = p.GetPath().pathString
            if p.GetName() == name and "Robotiq" in sp and "/Joints/" not in sp:
                out.append(p)
        return out

    added = 0
    for base in find_all("base_link"):
        gripper_root = base.GetParent()
        joints_scope = gripper_root.GetPath().AppendChild("Joints")
        T0 = xc.GetLocalToWorldTransform(base)
        # match each base_link to the inner_knuckles in the SAME gripper subtree
        prefix = gripper_root.GetPath().pathString
        for side, y in (("left", -abs(pivot_y)), ("right", abs(pivot_y))):
            kn = next((p for p in find_all(f"{side}_inner_knuckle")
                       if p.GetPath().pathString.startswith(prefix)), None)
            if kn is None:
                continue
            # AXIS ORDER differs between the two Robotiq USDs. trc-rollout targets the
            # Nucleus asset, whose base_link frame has the gripper axis along +Z, so it
            # writes (0, y, pivot_z). RoboLab's flattened asset puts the gripper axis
            # along +X -- read off its own finger_joint, whose localPos0 in base_link is
            # [0.05466, +/-0.0306, 0]. Using the Nucleus order here authors the constraint
            # ~6 cm from the real hinge and JAMS the linkage (finger_joint frozen ~0.04).
            w_base = Gf.Vec3d(pivot_z, y, 0.0) if x_axis_forward else Gf.Vec3d(0.0, y, pivot_z)
            w_world = T0.Transform(w_base)
            local1 = xc.GetLocalToWorldTransform(kn).GetInverse().Transform(w_world)
            jp = joints_scope.AppendChild(f"loop_{side}_inner")
            j = UsdPhysics.Joint.Define(stage, jp)
            j.CreateBody0Rel().SetTargets([base.GetPath()])
            j.CreateBody1Rel().SetTargets([kn.GetPath()])
            j.CreateLocalPos0Attr(Gf.Vec3f(w_base))
            j.CreateLocalPos1Attr(Gf.Vec3f(local1))
            j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
            j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
            j.CreateExcludeFromArticulationAttr(True)    # maximal-coord loop joint
            for ax in ("transX", "transY", "transZ"):    # low>high => DOF locked; rot free
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), ax)
                lim.CreateLowAttr(1.0)
                lim.CreateHighAttr(-1.0)
            added += 1
    return added
