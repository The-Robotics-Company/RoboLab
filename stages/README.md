# Inspection stages (TRC)

Standalone USD stages that compose a RoboLab task scene with a robot and its cameras at nominal poses, for visual
inspection in Isaac Sim. They are not RoboLab task scenes: the rollout harness spawns robots and cameras from the
Cfgs in `robolab/robots/` and `robolab/registrations/`; these files only mirror that layout so it can be looked at.
All references are relative (`../assets/...`), so keep this folder at the repo root.

| Stage | Robot | Cameras |
|---|---|---|
| `piperx_rubiks_cube_bowl.usda` | Piper-X (`assets/robots/piper_x/`), home pose | trc-spaces DC1 wrist + exo, 624x352 |
| `droid_rubiks_cube_bowl.usda` | Franka + Robotiq 2F-85 (stock RoboLab), DROID init pose | DROID wrist + over-shoulder-left 1280x720, egocentric viewport 864x480 |
| `droid_rubiks_cube_bowl_v0.usda` | same DROID robot on the franka_table, on the bare video-reconstructed `assets/scenes/rubiks_cube_bowl_v0/` scene | same DROID cameras |
| `droid_mugs4_measuringcup_drill_bowl.usda` | DROID robot on the RoboLab PickDrill scene (drill, 4 mugs, bowl, measuring cup) | same DROID cameras |
| `droid_mugs4_measuringcup_drill_bowl_v0.usda` | DROID robot on the bare video-reconstructed `assets/scenes/mugs4_measuringcup_drill_bowl_v0/` scene | same DROID cameras |

## Rig: one USD for the look, shared by previews and rollouts

`rigs/home_office.usda` holds everything about appearance that is not a task object: RoboLab's
`assets/backgrounds/default/home_office.exr` as a DomeLight (intensity 500, visible to the camera, the only light) and a
visual-only GroundPlane at z = -0.697. Both stages reference it, and `robolab/registrations/rig.py::rig_cfg("home_office")`
spawns the same file once at `/World/rig` in every registered env, so a rollout is lit by the identical prims. It is the
default for both robots in `policies/pi0_family/run_rollout.py` (`--rig home_office`; `--rig <other>.usda` for another
look, `--rig stock` for RoboLab's own per-robot lighting/background cfgs).

The ground plane is authored at the rig's own origin, so ONE rig file serves every scene: the stage (or `rig_cfg`)
places the rig prim at that scene's ground height. Heights in use: `rubiks_cube_bowl` -0.697, `mugs4_measuringcup_drill_bowl`
-0.650, `rubiks_cube_bowl_v0` -0.945, `mugs4_measuringcup_drill_bowl_v0` -0.8184. The dome is infinite, so moving the rig
moves only the ground. `run_rollout.py --rig-ground auto` (the default) reads the height off the task scene's own
`/GroundPlane`, so a rollout's visible ground always sits on the scene's collider ground.

## v0 scenes (video reconstructions)

Converted with `stages/convert_v0_scene.py <src_dir> <out_name> obj_000000=<name> ...` (the delivered USD does not label
its objects; identify them from `scene_info.txt` masses/static flags, the albedo textures and, when the delivery is in the
GT frame, object positions). Two are in the repo: `rubiks_cube_bowl_v0` (3 objects) and `mugs4_measuringcup_drill_bowl_v0`
(8 objects, PickDrillTask). The script renames the prims, drops room/lights/PhysicsScene, authors the franka_table mount
and a collider ground at the room floor, re-authors xforms as translate/orient/scale, tunes the convex decomposition and
authors mesh extents.

`assets/scenes/rubiks_cube_bowl_v0/rubiks_cube_bowl_v0.usda` is the bare form of a video-reconstructed scene: only the
essential assets (table as kinematic rigid body, bowl and rubiks_cube as dynamic rigid bodies, named as RubiksCubeTask
expects, the franka_table robot mount as in every RoboLab scene, plus an invisible collider ground). The mount is 0.795 m
tall while the video floor is at -0.945, so it hangs 15 cm above the visible ground; rollouts place it the same way. Room, lights, PhysicsScene and room materials were stripped; light, backdrop
and visible ground are added in one step by the rig. Conversion script: `stages/convert_v0_scene.py` (Sdf.CopySpec of the
object prims, references retargeted to `./meshes/`). Rollout: `run_rollout.py --robot droid --task RubiksCubeTask
--scene-variant v0 --rig home_office_floor945`. Do not size these objects from a world-space bounding box: the meshes' local frames are rotated relative to the world
and their local boxes are not tight, so `BBoxCache` reports the box of a rotated box (cube 14.5 cm, bowl 18 cm). Measured
from the vertices the cube is ~6-7.6 cm across (GT 5.8 cm) and the bowl 13.6 cm wide, ~9 cm tall (GT 16.1 x 5.5 cm), i.e.
close to the real objects. The mesh files had no authored `extent`; one was added (computed from the vertices).

Colliders stay derived from the visual meshes (convex decomposition), tuned in the scene layer to follow the mesh
faithfully: table and bowl use `maxConvexHulls` 128, `voxelResolution` 4e6, `errorPercentage` 1.0 / 0.5, `shrinkWrap` on
(PhysX defaults: 32 hulls, 5e5 voxels, 10 %). The reconstructed tabletop itself is wavy (13 mm std, ~4 cm peak to peak,
29 mm/m tilt) and the collider reproduces that by design. Measured effect vs the delivered defaults: objects rest 12-16 mm
lower (the default hulls bulged above the visual surface, so objects floated); a cube dropped into the bowl lands inside
the cavity with both settings. Check script: `stages/physics_check.py <stage.usda>` (settle 3 s, drop cube into bowl 4 s).

Object xforms must be standard `translate` / `orient` / `scale` ops: RoboLab rejects a prim carrying a bare
`xformOp:transform` matrix ("not a xformable prim with standard transform operations"). The conversion script decomposes
the reconstruction's matrices (uniform scale, no shear) into those ops; world placement is unchanged.

Verified 2026-09-09: `run_rollout.py --robot droid --task RubiksCubeTask --scene-variant v0 --rig home_office_floor945`,
pi05_droid_jointpos, 10 episodes: 10/10 successes (7-16 s), videos on W&B piperx-robolab/biwoi7hu. Caveat on the success
predicate: the reconstructed bowl's local frame is tilted 112 deg from world up, so RoboLab's open-top containment box is
rotated with it (still correct for objects resting in the bowl, but the tipped-container check is meaningless and recorded
object poses are in these tilted frames). Upright, identity-aligned local frames at export would fix that.

Why not put it in the scene USD: RoboLab scrapes scene USDs into per-object asset cfgs (lights are dropped) and clones
the scene per env, so a dome inside the scene would be lost or stacked N times. Global prims must be spawned once at
/World, which is what the rig cfg does. The ground height is authored for the RoboLab table scenes (z = -0.697); a scene
with a different ground needs its own rig file.

Both stages also carry three colleague render presets (`his_angled`, `his_front`, `his_top`; look-at definitions in
`colleague_camera_presets.json`).

## Tools (run with `OMNI_KIT_ACCEPT_EULA=YES /opt/IsaacSim/python.sh <tool>`)

| Tool | What |
|---|---|
| `render_cams.py <stage> <out_dir> --cam <prim> ... [--res WxH] [--closeup <prim>] [--no-lights]` | headless render from named camera prims plus auto overview views, after settling physics |
| `render_views.py <stage> <out_dir> <views.json> [--res 640x480 --focal 24 --haperture 20.955]` | headless render from look-at views with given intrinsics |
| `overview_views_drill.json` | the same for the PickDrill scene pair (their content bbox differs) |
| `overview_views.json` | fixed overview poses (angled / robot side / top, aimed at the GT stage's content centre) so different stages render from identical cameras: `render_views.py <stage> <out> overview_views.json --res 1280x720` |
| `pack_stage.py <stage> <repo_root> <out_dir>` | self-contained bundle of a stage and every dependency (for S3) |
| `open_scene.py` | Isaac Sim GUI startup script: `ROBOLAB_SCENE=<stage> isaac-sim.sh --no-ros-env --exec open_scene.py` |
| `stl_to_usd.py` | mm STL -> metre USD mesh (used for the Piper-X wrist-cam mount) |

Packed bundles with preview renders live at `s3://piperx-pick-cube-cup/assets/isaac/{piperx,droid}_rubiks_cube_bowl/`;
the matching robot config bundles at `.../{piperx,droid}_robolab_cfg/`.
