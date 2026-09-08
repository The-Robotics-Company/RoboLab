# Piper-X (AgileX PiPER-X 6-DOF arm + parallel gripper)

Copied 2026-09-07 from `trc-spaces/external/piper-x-arm/assets/piper_x/usd/` (piper-x-arm repo).
Converted from `piper_x.urdf` with IsaacLab's UrdfConverter (see `config.yaml`): fixed base,
convex-hull colliders, fixed joints not merged.

Physics layer is the converter's untouched output: position drives kp 100 / kv 10 on every joint (USD angular gains
are stored per degree), max force 100 Nm arm / 10 N fingers, gravity on, no physics material, solver 32/1.
Control-relevant physics (actuator gains and limits, gravity compensation, finger friction, init state, gripper
mapping, cameras, control rate) is set in the RoboLab robot Cfg, which overrides the USD drives at spawn. The MuJoCo
reference values are: arm kp 150 / kv 15 / 100 Nm, wrist kp 60 / kv 6 / 30 Nm, fingers kp 200 / kv 20 / 40 N,
gravcomp on every body, friction 1.0 (`trc-spaces/assets/piper_x/piper_x.xml`).

Prims: `/piper_x/{base_link, link1..link6, gripper_base, gripper_link1, gripper_link2}`,
joints under `/piper_x/joints/{joint1..joint6, gripper_base_joint, gripper_joint1, gripper_joint2}`,
articulation root on `/piper_x/root_joint`. Revolute limits are stored in degrees.

Nominal setup (from trc-spaces `assets/piper_x/piper_x.xml` and `asset_library/cubes_in_cup_scene.xml`):
- base_link at the env origin, identity orientation, mounted on the table-top plane z = 0
- home joint pose: joint1..6 = [0, 1.2, -1.2, 0, 0, 0] rad; gripper_joint1 = +0.035 m, gripper_joint2 = -0.035 m (open; RoboLab cfg GRIPPER_OPEN, changed from 0.025 on 2026-09-08 so the 58 mm rubiks_cube fits)
- wrist camera (Orbbec DC1) on `gripper_base`: pos (-0.07734, -0.008, 0.04364), quat wxyz (0.190415, 0.680986, -0.680986, -0.190415), fovy 52.5 deg
- exo camera, world-fixed: pos (1.1717, -0.31, 0.5726), quat wxyz (0.690853, 0.44014, 0.308196, 0.483751); same DC1 intrinsics as the wrist camera (fovy 52.5 deg). NOTE: trc-spaces cubes_in_cup_scene.xml still says fovy 45 for exo_camera
- render resolution 624x352
- wrist-cam mount bracket: wrist_cam_mount.usd (from piper-x-arm cad/models/arm/wrist_cam_mount.stl, mm->m), placed at (0, 0, 0.005) in gripper_base, visual only
- DC1 body: no CAD model exists (Orbbec forum thread 4341: DaBai is "not an officially released product"; Orbbec ROS2 description only has DaBai DCW2 / Max Pro). Modelled as a box 80 x 23 x 22 mm behind the lens (dims from cad/models/arm/preview_mount.py)
- Intrinsics -> USD: focalLength 20 mm, verticalAperture = 2*20*tan(52.5/2) = 19.7258 mm, horizontalAperture = 19.7258*624/352 = 34.9685 mm (fovx 82.3 deg)
