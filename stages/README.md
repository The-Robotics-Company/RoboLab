# Inspection stages (TRC)

Standalone USD stages that compose a RoboLab task scene with a robot and its cameras at nominal poses, for visual
inspection in Isaac Sim. They are not RoboLab task scenes: the rollout harness spawns robots and cameras from the
Cfgs in `robolab/robots/` and `robolab/registrations/`; these files only mirror that layout so it can be looked at.
All references are relative (`../assets/...`), so keep this folder at the repo root.

| Stage | Robot | Cameras |
|---|---|---|
| `piperx_rubiks_cube_bowl.usda` | Piper-X (`assets/robots/piper_x/`), home pose | trc-spaces DC1 wrist + exo, 624x352 |
| `droid_rubiks_cube_bowl.usda` | Franka + Robotiq 2F-85 (stock RoboLab), DROID init pose | DROID wrist + over-shoulder-left 1280x720, egocentric viewport 864x480 |

Both use the same rig: `assets/backgrounds/default/home_office.exr` as a DomeLight (intensity 500, visible to the
camera) as the only light, a visual-only GroundPlane at z = -0.697, and three colleague render presets
(`his_angled`, `his_front`, `his_top`; look-at definitions in `colleague_camera_presets.json`).

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
