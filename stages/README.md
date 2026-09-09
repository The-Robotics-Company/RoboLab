# Inspection stages (TRC)

Standalone USD stages that compose a RoboLab task scene with a robot and its cameras at nominal poses, for visual
inspection in Isaac Sim. They are not RoboLab task scenes: the rollout harness spawns robots and cameras from the
Cfgs in `robolab/robots/` and `robolab/registrations/`; these files only mirror that layout so it can be looked at.
All references are relative (`../assets/...`), so keep this folder at the repo root.

| Stage | Robot | Cameras |
|---|---|---|
| `piperx_rubiks_cube_bowl.usda` | Piper-X (`assets/robots/piper_x/`), home pose | trc-spaces DC1 wrist + exo, 624x352 |
| `droid_rubiks_cube_bowl.usda` | Franka + Robotiq 2F-85 (stock RoboLab), DROID init pose | DROID wrist + over-shoulder-left 1280x720, egocentric viewport 864x480 |
| `droid_rubiks_cube_bowl_v0.usda` | same DROID robot, no robot table, on the bare video-reconstructed `assets/scenes/rubiks_cube_bowl_v0/` scene; rig `home_office_floor945` | same DROID cameras |

## Rig: one USD for the look, shared by previews and rollouts

`rigs/home_office.usda` holds everything about appearance that is not a task object: RoboLab's
`assets/backgrounds/default/home_office.exr` as a DomeLight (intensity 500, visible to the camera, the only light) and a
visual-only GroundPlane at z = -0.697. Both stages reference it, and `robolab/registrations/rig.py::rig_cfg("home_office")`
spawns the same file once at `/World/rig` in every registered env, so a rollout is lit by the identical prims. It is the
default for both robots in `policies/pi0_family/run_rollout.py` (`--rig home_office`; `--rig <other>.usda` for another
look, `--rig stock` for RoboLab's own per-robot lighting/background cfgs).

`rigs/home_office_floor945.usda` is the same rig with the ground at z = -0.945, the floor height of the video-reconstructed
scenes (`rubiks_cube_bowl_v0`). Pick the rig whose ground matches the scene's collider ground.

## v0 scenes (video reconstructions)

`assets/scenes/rubiks_cube_bowl_v0/rubiks_cube_bowl_v0.usda` is the bare form of a video-reconstructed scene: only the
essential assets (table as kinematic rigid body, bowl and rubiks_cube as dynamic rigid bodies, named as RubiksCubeTask
expects, plus an invisible collider ground). Room, lights, PhysicsScene and room materials were stripped; light, backdrop
and visible ground are added in one step by the rig. Conversion script: `stages/convert_v0_scene.py` (Sdf.CopySpec of the
object prims, references retargeted to `./meshes/`). Rollout: `run_rollout.py --robot droid --task RubiksCubeTask
--scene-variant v0 --rig home_office_floor945`. Note the delivered object scale (cube 14 cm, bowl 17 cm tall) is larger
than the real objects; the Robotiq 2F-85 cannot grasp the cube as delivered.

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
| `pack_stage.py <stage> <repo_root> <out_dir>` | self-contained bundle of a stage and every dependency (for S3) |
| `open_scene.py` | Isaac Sim GUI startup script: `ROBOLAB_SCENE=<stage> isaac-sim.sh --no-ros-env --exec open_scene.py` |
| `stl_to_usd.py` | mm STL -> metre USD mesh (used for the Piper-X wrist-cam mount) |

Packed bundles with preview renders live at `s3://piperx-pick-cube-cup/assets/isaac/{piperx,droid}_rubiks_cube_bowl/`;
the matching robot config bundles at `.../{piperx,droid}_robolab_cfg/`.
