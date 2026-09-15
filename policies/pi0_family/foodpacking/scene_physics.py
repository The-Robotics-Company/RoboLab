# SPDX-License-Identifier: Apache-2.0
"""One physics setup for food_packing, shared by the demo generator and the eval harness.

Train and test MUST run the same scene. Everything here was found the hard way and each
knob is a measured fix, so it lives in one place instead of drifting between scripts.

Works on a single-env stage (/World/SceneObjects/<obj>) and on a cloned one
(/World/envs/env_N/SceneObjects/<obj>): it matches on the "SceneObjects" path segment
and the prim name. Call AFTER the stage is built and BEFORE sim.reset().
"""
GRASPABLES = ("mustard_bottle", "spam_can")
BINS = ("grey_bin_left", "grey_bin_right")
# Never manipulated. The soup_can stands on the small_box next to the container: a
# marginally stable, close-packed stack that PhysX blows apart at spawn in ~1/16 envs
# (bit-reproducible per config, objects at 3-5 m/s before the arm has moved).
DISTRACTORS = ("cheezit_box", "soup_can", "sugar_box", "container", "small_box")
ALL_SCENE_OBJS = GRASPABLES + DISTRACTORS + BINS


def apply_scene_physics(stage, *, pin_bins=True, pin_distractors=False,
                        bin_collider="convexDecomposition", contact_tune=True,
                        depen_cap=0.5, loop_closure=True, x_axis_forward=False,
                        distractor_tune=False, obj_vel_cap=0.0,
                        distractor_collider="keep", graspable_collider="keep", set_mass=()):
    """Author every physics override on `stage`. Returns a dict of counts for the log."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from robotiq_loop import author_robotiq_loop_closure

    pads = bins_pinned = dist_pinned = bin_meshes = recollided = 0
    masses = dict(kv.split("=") for kv in set_mass) if set_mass else {}
    # The delivery gives EVERY object a 64-hull convex decomposition (128 verts/hull). For
    # boxes and cans a single hull is exact and far more stable: hull-hull chatter between
    # dozens of tiny sub-hulls is how a 450 g can sinks into a 60 g box and then explodes.
    recollide = {}
    if distractor_collider != "keep":
        recollide.update({n: distractor_collider for n in DISTRACTORS})
    if graspable_collider != "keep":
        recollide.update({n: graspable_collider for n in GRASPABLES})
    fric = None
    if contact_tune:
        UsdGeom.Scope.Define(stage, "/World/Materials")
        fric = UsdShade.Material.Define(stage, "/World/Materials/HighFriction")
        pm = UsdPhysics.MaterialAPI.Apply(fric.GetPrim())
        pm.CreateStaticFrictionAttr(1.5); pm.CreateDynamicFrictionAttr(1.5); pm.CreateRestitutionAttr(0.0)

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        nm = prim.GetName()
        if fric is not None and "inner_finger" in nm.lower() and "knuckle" not in nm.lower():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                fric, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
            pads += 1
        if "SceneObjects" not in path:
            continue
        if nm in ALL_SCENE_OBJS:
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            # Cap how fast PhysX may push interpenetrating bodies apart. Uncapped it
            # resolves overlap explosively ("objects fly off the table"). RoboLab caps the
            # robot at 5.0; the delivery's objects had no cap and sit up to 8 mm into the
            # tabletop. 0.5 m/s resolves overlap without launching anything.
            prb.CreateMaxDepenetrationVelocityAttr(depen_cap)
            # Receptacles are not manipulated, but ship as 0.9 kg dynamic bodies. A
            # position-controlled arm (stiffness 400, 87 Nm) that clips a rim drives
            # through and launches the bin (measured 562 m away). trc-rollout pins its
            # receptacle the same way.
            if nm in BINS and pin_bins:
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True); bins_pinned += 1
            if nm in DISTRACTORS and pin_distractors:
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True); dist_pinned += 1
            if obj_vel_cap > 0:
                # A blow-up cannot fling anything faster than this; the stack may still
                # topple but it stays on the table and settles.
                prb.CreateMaxLinearVelocityAttr(obj_vel_cap)
                prb.CreateMaxAngularVelocityAttr(10.0 * obj_vel_cap)
            if nm in DISTRACTORS and distractor_tune:
                prb.CreateSolverPositionIterationCountAttr(64)
                prb.CreateSolverVelocityIterationCountAttr(8)
                prb.CreateLinearDampingAttr(2.0)
                prb.CreateAngularDampingAttr(2.0)
            if nm in masses:
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(masses[nm]))
            if nm in GRASPABLES and contact_tune:
                prb.CreateSolverPositionIterationCountAttr(64)
                prb.CreateSolverVelocityIterationCountAttr(8)
        # The two bins are the only sdf colliders in the delivery (the rest are
        # convexDecomposition, the table an exact mesh) and the only two that ever
        # exploded; sdf cooking also wedged multi-env startup.
        for n, approx in recollide.items():
            if f"/{n}/" in path or path.endswith(f"/{n}"):
                mc = UsdPhysics.MeshCollisionAPI(prim)
                if mc and mc.GetApproximationAttr().Get():
                    mc.GetApproximationAttr().Set(approx); recollided += 1
        if bin_collider != "sdf" and any(b in path for b in BINS):
            mc = UsdPhysics.MeshCollisionAPI(prim)
            if mc and mc.GetApproximationAttr().Get():
                mc.GetApproximationAttr().Set(bin_collider); bin_meshes += 1

    # Robotiq 2F-85 four-bar loop the USD leaves open. Without it the stroke is a
    # distorted 8-70 mm and nothing can be grasped (0/16).
    nloop = author_robotiq_loop_closure(x_axis_forward=x_axis_forward) if loop_closure else 0
    return dict(pads=pads, loop_joints=nloop, bins_pinned=bins_pinned,
                distractors_pinned=dist_pinned, bin_meshes=bin_meshes, bin_collider=bin_collider,
                distractor_tune=distractor_tune, obj_vel_cap=obj_vel_cap,
                recollided=recollided, masses=masses)


def add_physics_args(p):
    """The same flags, same defaults, on both scripts."""
    p.add_argument("--pin-bins", type=int, default=1, help="1 = bins kinematic (cannot be launched)")
    p.add_argument("--pin-distractors", type=int, default=0,
                   help="1 = the five never-manipulated objects kinematic. Decision: everything stays dynamic.")
    p.add_argument("--distractor-tune", type=int, default=0,
                   help="1 = 64/8 solver iters + damping 2.0 on the five distractors")
    p.add_argument("--obj-vel-cap", type=float, default=0.0, help="max linear velocity (m/s) on all scene objects, 0 = off")
    # convexHull is THE fix for the distractor stack: 16/16 clean vs 10/16 with the delivered
    # 64-hull decomposition (E1 vs E5, 2026-09-14). Everything stays dynamic.
    p.add_argument("--distractor-collider", default="convexHull", choices=["keep", "convexHull", "convexDecomposition", "boundingCube"])
    p.add_argument("--graspable-collider", default="keep", choices=["keep", "convexHull", "convexDecomposition"])
    p.add_argument("--set-mass", nargs="*", default=[], help="name=kg overrides, e.g. small_box=0.3")
    p.add_argument("--scene", default="opus_gt", choices=sorted(SCENES),
                   help="which table+objects to load (see SCENES). Train and eval MUST use the same one")
    p.add_argument("--wrist-cam", default="legacy", choices=sorted(WRIST_CAM),
                   help="wrist camera preset (see WRIST_CAM). Train and eval MUST use the same one")
    p.add_argument("--pre-settle", type=int, default=0,
                   help="physics steps to run once after reset before capturing the nominal poses (bakes a rested layout)")
    p.add_argument("--bin-collider", default="convexDecomposition", choices=["sdf", "convexDecomposition"])
    p.add_argument("--contact-tune", type=int, default=1,
                   help="1 = friction-1.5 finger pads + 64/8 solver iters on the two graspables")
    p.add_argument("--loop-closure", type=int, default=1, help="1 = Robotiq four-bar loop-closure D6 joints")
    p.add_argument("--depen-cap", type=float, default=0.5, help="max depenetration velocity, m/s, all scene objects")
    return p


# Wrist camera presets, relative to Robotiq_2F_85/base_link.
#  legacy: RoboLab's DroidCfg offset, authored for the flattened asset. On the Nucleus
#          variant it sits INSIDE the housing: ~80% of the image is gripper body.
#  side:   7 cm out on +X, looking along +Z (toward the fingertips) tilted 25 deg inward,
#          gripper at the image bottom -- fingers and workspace both visible (probe_wrist_cam).
WRIST_CAM = {
    "legacy": dict(pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"),
    "side":   dict(pos=(0.07, 0.0, -0.02), rot=(0.0000, 0.5360, 0.0000, 0.8442), convention="world"),
}


def reap_zombie_children():
    """Kit spawns two helpers (Omniverse Hub, nvidia-ngx-updater) that exit at once and are
    never waited on, so every Isaac process shows two <defunct> children for its whole life.
    Reap ONLY children already in state Z: nothing else can be waiting on those."""
    import os
    me, n = os.getpid(), 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            if rest[0] == "Z" and int(rest[1]) == me:
                os.waitpid(int(d), os.WNOHANG); n += 1
        except (OSError, IndexError, ValueError, ChildProcessError):
            pass
    return n


# Scenes this pipeline can run. Same robot, cameras, DR file, physics flags and success test;
# only the table + objects differ. Train and eval MUST use the same one.
#  opus_gt: the Robolab delivery reconstruction (what every demo / result so far is on)
#  gt:      RoboLab's authored food_packing.usda (ycb + vomp assets), prims renamed to the
#           canonical names -- see assets/scenes/food_packing_gt_canon.usda
SCENES = {
    "opus_gt": "food_packing_opus_gt.usda",
    "gt": "food_packing_gt_canon.usda",
}

# Bin geometry from mesh vertices, NOT from authored extents (the delivery's are stale). Per
# bin: geometric-centre XY offset from the body root, INNER XY half-extents (outer minus
# ~1 cm wall) and the rim height above the root. The old rule "centre within 13 cm of the
# root" rejected a bottle standing in a corner (~14 cm out) and passed one lying across the rim.
BIN_GEOM = {
    "opus_gt": {
        "grey_bin_left":  ((-0.014, -0.012), (0.144, 0.128), 0.10),   # outer 0.308 x 0.275, rim z +0.095
        "grey_bin_right": ((-0.012, +0.002), (0.126, 0.119), 0.10),   # outer 0.271 x 0.257, rim z +0.120
    },
    # vomp bins, mesh vertices x scene scale (0.7 / 1.4), root at the mesh centre, 180 deg about Z.
    # Both are open-front stacking bins: the low lip faces the robot; rim = full wall height.
    "gt": {
        "grey_bin_left":  ((0.0, 0.0), (0.168, 0.095), 0.131),        # bin_a06: outer 0.356 x 0.210, rim +0.131 (lip +0.104)
        "grey_bin_right": ((0.0, 0.0), (0.152, 0.088), 0.168),        # bin_b03: outer 0.324 x 0.196, rim +0.168 (lip +0.069)
    },
}
BIN_XY = {k: v[:2] for k, v in BIN_GEOM["opus_gt"].items()}
BIN_Z_MAX = {k: v[2] for k, v in BIN_GEOM["opus_gt"].items()}


def select_scene(name):
    """Point in_bin at `name`'s bin geometry. Call once, right after parsing args."""
    global BIN_XY, BIN_Z_MAX
    BIN_XY = {k: v[:2] for k, v in BIN_GEOM[name].items()}
    BIN_Z_MAX = {k: v[2] for k, v in BIN_GEOM[name].items()}
    return SCENES[name]


def in_bin(obj_xyz, bin_xyz, bin_name):
    """(ok, why): object centre inside the bin's inner footprint and below the rim band.
    Bins are kinematic and axis-aligned in this scene, so a world-axis box test is exact."""
    off, half = BIN_XY[bin_name]
    zmax = BIN_Z_MAX[bin_name]
    dx = float(obj_xyz[0] - (bin_xyz[0] + off[0]))
    dy = float(obj_xyz[1] - (bin_xyz[1] + off[1]))
    dz = float(obj_xyz[2] - bin_xyz[2])
    ok = abs(dx) < half[0] and abs(dy) < half[1] and dz < zmax
    return ok, f"dx={dx:+.3f}/{half[0]:.3f} dy={dy:+.3f}/{half[1]:.3f} dz={dz:+.3f}/<{zmax:.2f}"
