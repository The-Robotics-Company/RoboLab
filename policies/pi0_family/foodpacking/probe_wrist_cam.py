# SPDX-License-Identifier: Apache-2.0
"""Render candidate wrist-camera poses on the Nucleus Franka+Robotiq, one Isaac boot.
The delivered offset (from RoboLab's flattened asset) looks into the gripper body on this
asset. Six axis-aligned 'world'-convention orientations, each saved as a PNG."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--out", default="/tmp/wrist_probe")
p.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
p.add_argument("--cands", default="", help="'label:px,py,pz:fx,fy,fz;...' camera position + forward dir in base_link frame; "
               "image-up = away from the gripper centre line. Replaces the 6 axis defaults when given")
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args(); a.enable_cameras = True
app = AppLauncher(a).app
import os, math, numpy as np, cv2, torch                                     # noqa: E402
import isaaclab.sim as sim_utils                                             # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext                    # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg                     # noqa: E402
from isaaclab.assets import Articulation                                     # noqa: E402
from isaaclab_assets import FRANKA_ROBOTIQ_GRIPPER_CFG                       # noqa: E402
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_cfg import SCENE  # noqa: E402
HOME = [0.0, 0.0, 0.0, -1.178, 0.0, 1.08, 0.0]

def q_axis(axis, deg):
    s = math.sin(math.radians(deg) / 2); c = math.cos(math.radians(deg) / 2)
    v = {"x": (s, 0, 0), "y": (0, s, 0), "z": (0, 0, s)}[axis]
    return (c, *v)

def main():
    sim = SimulationContext(SimulationCfg(dt=1 / 120, device=a.device))
    sim_utils.DomeLightCfg(intensity=900.0).func("/World/Dome", sim_utils.DomeLightCfg(intensity=900.0))
    sim_utils.UsdFileCfg(usd_path=SCENE).func("/World/SceneObjects", sim_utils.UsdFileCfg(usd_path=SCENE))
    cfg = FRANKA_ROBOTIQ_GRIPPER_CFG.copy(); cfg.prim_path = "/World/robot"
    cfg.init_state.pos = (0.0, 0.0, 0.0); cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot = Articulation(cfg)
    cands = {"px": (1, 0, 0, 0), "nx": q_axis("z", 180), "py": q_axis("z", 90), "ny": q_axis("z", -90),
             "pz": q_axis("y", -90), "nz": q_axis("y", 90)}
    poses = {lab: (tuple(a.pos), q) for lab, q in cands.items()}
    if a.cands:
        from scipy.spatial.transform import Rotation as _R
        poses = {}
        for item in a.cands.split(";"):
            lab, ps, fs = item.split(":")
            pos = np.array([float(v) for v in ps.split(",")]); f = np.array([float(v) for v in fs.split(",")]); f /= np.linalg.norm(f)
            u = -pos.copy(); u[2] = 0.0                      # 'up' points from camera toward the centre line
            if np.linalg.norm(u) < 1e-6: u = np.array([0, 1, 0.0])
            u = u - (u @ f) * f; u /= np.linalg.norm(u)
            left = np.cross(u, f)                              # world convention: x=forward, y=left, z=up
            R = np.stack([f, left, u], 1)
            x, y, z, w = _R.from_matrix(R).as_quat()
            poses[lab] = (tuple(pos), (w, x, y, z))
    cams = {}
    for lab, (pos, q) in poses.items():
        cams[lab] = TiledCamera(TiledCameraCfg(
            prim_path=f"/World/robot/Robotiq_2F_85_edit/Robotiq_2F_85/base_link/probe_{lab}",
            height=180, width=320, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=2.8, focus_distance=28.0,
                                             horizontal_aperture=5.376, vertical_aperture=3.024),
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=q, convention="world")))
    sim.reset()
    q0 = robot.data.default_joint_pos.clone()
    jn = list(robot.data.joint_names); arm = [jn.index(f"panda_joint{i}") for i in range(1, 8)]
    q0[0, arm] = torch.tensor(HOME, device=q0.device)
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0)); robot.set_joint_position_target(q0); robot.write_data_to_sim()
    for _ in range(60):
        sim.step(render=False)
    robot.update(1 / 15); sim.render()
    os.makedirs(a.out, exist_ok=True)
    for lab, cam in cams.items():
        cam.update(0.0)
        img = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        if img.dtype != np.uint8:
            img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
        cv2.putText(img, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{a.out}/wrist_{lab}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[probe] wrote {len(cams)} candidates -> {a.out}", flush=True)

main()
import sys, os as _os
sys.stdout.flush(); _os._exit(0)
