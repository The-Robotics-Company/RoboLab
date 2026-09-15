# SPDX-License-Identifier: Apache-2.0
"""Vectorised expert demo generation for food_packing -- N envs stepped together.

  * InteractiveScene(num_envs=N, replicate_physics=True) clones the scene once; TiledCamera
    renders ALL envs in one pass at the record resolution (320x180).
  * One RmpFlow per env (lula is CPU-side); every env runs the same waypoint script
    (pick_sequence.py) in lockstep, with per-env goals from its own DR draw.
  * Waypoint budgets are CAPS: a phase ends as soon as every healthy env has stopped moving.
  * Physics, success test and cameras come from scene_physics.py / scene_cfg.py, shared with
    eval.py so train and test are the same scene by construction.

    python gen_demos.py --num-envs 16 --episodes 160 --dr-start 20 --out ~/datasets/food_packing_demos
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_physics import apply_scene_physics, add_physics_args, ALL_SCENE_OBJS, in_bin, reap_zombie_children, select_scene, WRIST_CAM  # noqa: E402
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=16)
p.add_argument("--episodes", type=int, default=64, help="total demos attempted (rounded up to whole batches)")
p.add_argument("--dr-start", type=int, default=20)
p.add_argument("--dr-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dr_episodes.json"))
p.add_argument("--grasp-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasps"),
               help="object-frame grasps recorded with trc-rollout (examples/food_packing_pick.py --record-grasp)")
p.add_argument("--out", default=os.path.expanduser("~/datasets/food_packing_demos"))
p.add_argument("--env-spacing", type=float, default=14.0,
               help="metres between envs. The exo camera is wide-angle (2.1 mm focal), so a\n                     close neighbour lands in frame and pollutes the policy view.")
p.add_argument("--dev", action="store_true",
               help="boot once, then reload pick_sequence.py on save (no ~2 min reboot "
                    "per waypoint tweak). Same pattern as trc-rollout --dev / pointbody dev.py.")
add_physics_args(p)
p.add_argument("--video", default="", help="write a tiled mp4 of every env")
p.add_argument("--video-fps", type=int, default=15)
p.add_argument("--prompt", default="put the mustard bottle in the left bin and the spam can in the right bin")
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
select_scene(a.scene)
a.enable_cameras = True
app = AppLauncher(a).app

import numpy as np, torch                                              # noqa: E402
from scipy.spatial.transform import Rotation                           # noqa: E402
import isaaclab.sim as sim_utils                                       # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext              # noqa: E402

HZ, DECIM = 15, 8
ORDER = [("mustard_bottle", "grey_bin_left"), ("spam_can", "grey_bin_right")]
OPEN, CLOSE = 0.0, float(np.pi / 4)
# (label, budget) sized from measured single-env convergence
PHASES = [("align", 150), ("descend", 90), ("close", 70), ("lift", 90),
          ("transport", 150), ("release", 20), ("retract", 40)]


def q2m(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


from scene_cfg import build_scene, IMG_W, IMG_H  # noqa: E402  (shared with eval.py)


class ArmRMP:
    """One RmpFlow per env. lula is CPU-side, so N instances are cheap next to rendering."""
    def __init__(self, hand_offset_z, base_pos, base_quat):
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.robot_motion.motion_generation")
        import isaacsim.robot_motion.motion_generation as mg
        from isaacsim.robot_motion.motion_generation import RmpFlow
        from pathlib import Path
        d = Path(mg.__file__).parents[3] / "motion_policy_configs" / "franka"
        self.rmp = RmpFlow(robot_description_path=str(d / "rmpflow" / "robot_descriptor.yaml"),
                           urdf_path=str(d / "lula_franka_gen.urdf"),
                           rmpflow_config_path=str(d / "rmpflow" / "franka_rmpflow_common.yaml"),
                           end_effector_frame_name="panda_hand",
                           maximum_substep_size=1.0 / 240.0,
                           ignore_robot_state_updates=True)
        self.off = float(hand_offset_z)
        self.base_pos, self.base_quat = np.asarray(base_pos), np.asarray(base_quat)
        self.rmp.set_robot_base_pose(self.base_pos, self.base_quat)

    def reset(self):
        try: self.rmp.reset()
        except Exception: pass
        self.rmp.set_robot_base_pose(self.base_pos, self.base_quat)

    def set_goal(self, tcp_w, hand_quat):
        R = q2m(hand_quat)
        hand = np.asarray(tcp_w, float) - R[:, 2] * self.off
        q = Rotation.from_matrix(R).as_quat()
        self.rmp.set_end_effector_target(hand, np.array([q[3], q[0], q[1], q[2]]))

    def compute(self, q, qd, dt):
        tq, _ = self.rmp.compute_joint_targets(q, qd, q, qd, dt)
        return tq


def load_grasp(name, gdir):
    with open(os.path.join(gdir, f"{name}.json")) as f:
        d = json.load(f)["authored"]
    return np.asarray(d["position"], float), np.asarray(d["quat"], float)


def world_grasp(lp, lq, op, oq):
    R = q2m(oq)
    pos = np.asarray(op, float) + R @ lp
    q = (Rotation.from_matrix(R) * Rotation.from_quat([lq[1], lq[2], lq[3], lq[0]])).as_quat()
    return pos, np.array([q[3], q[0], q[1], q[2]])


import importlib
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick_sequence as _seq


def _reload_seq():
    """Reload the waypoint module, clearing the stale .pyc first.

    Two traps this avoids, both documented in trc-rollout's --dev loop: a cached
    .pyc can beat the edited source, and new modules need invalidate_caches().
    """
    global _seq
    if getattr(_seq, "__cached__", None):
        import pathlib
        pathlib.Path(_seq.__cached__).unlink(missing_ok=True)
    importlib.invalidate_caches()
    try:
        _seq = importlib.reload(_seq)
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        print("[dev] reload FAILED -- keeping previous waypoints", flush=True)
        return False


def main():
    N = a.num_envs
    sim = SimulationContext(SimulationCfg(dt=1.0 / (HZ * DECIM), device=a.device))
    scene = build_scene(N, a.env_spacing, a.wrist_cam, IMG_W, IMG_H, a.scene)

    import isaacsim.core.utils.stage as stage_utils                      # noqa: E402
    import sys as _sys

    st = stage_utils.get_current_stage()
    phys = apply_scene_physics(st, pin_bins=a.pin_bins, pin_distractors=a.pin_distractors,
                               bin_collider=a.bin_collider, contact_tune=a.contact_tune,
                               depen_cap=a.depen_cap, loop_closure=a.loop_closure,
                               distractor_tune=a.distractor_tune, obj_vel_cap=a.obj_vel_cap,
                               distractor_collider=a.distractor_collider,
                               graspable_collider=a.graspable_collider, set_mass=a.set_mass)
    print(f"[vec] {N} envs | {phys}", flush=True)

    sim.reset()
    # Optional pre-settle: the delivered poses interpenetrate by mm; letting the scene rest
    # once and taking THAT as nominal removes the spawn-time depenetration kick.
    for _ in range(a.pre_settle):
        sim.step(render=False)
    if a.pre_settle:
        scene.update(1.0 / HZ)
    robot = scene["robot"]
    exo, wrist = scene["exo"], scene["wrist"]
    ALL_OBJS = ("mustard_bottle", "spam_can", "grey_bin_left", "grey_bin_right",
                "cheezit_box", "soup_can", "sugar_box", "container", "small_box")
    objs = {n: scene[n] for n in ALL_OBJS}
    origins = scene.env_origins.cpu().numpy()
    _org = scene.env_origins
    import collections as _col
    _hist = _col.deque(maxlen=5)

    jn = list(robot.data.joint_names)
    arm = [jn.index(f"panda_joint{i}") for i in range(1, 8)]
    fid = jn.index("finger_joint")
    hand_idx = robot.data.body_names.index("panda_hand")
    q0 = robot.data.default_joint_pos.clone()
    nominal = {n: objs[n].data.root_state_w.clone() for n in objs}
    rmps = [ArmRMP(0.15, origins[i], np.array([1.0, 0, 0, 0])) for i in range(N)]
    grasps = {o: load_grasp(o, a.grasp_dir) for o, _ in ORDER}

    # Release just above the bin RIM, not a fixed 32 cm over the bin centre. A long
    # drop bounces the object out and shoves the bin, which is itself a dynamic body.
    # Same rule as trc-rollout: release_tcp_z = rim + (grasp_z - object_bottom) + 3 cm,
    # so the object's underside arrives 3 cm over the rim. Offsets from each body's
    # root are constant for a rigid shape, so measure them once.
    from pxr import Usd as _U2, UsdGeom as _UG2
    from pxr import Gf as _Gf2
    _st2 = stage_utils.get_current_stage()

    def _z_off(name):
        # From real vertices, NOT BBoxCache: the delivery's authored extents are the
        # pre-transform ones and read far too large -- they put a 12 cm bin's rim at
        # +0.19. Same trap the importer had to work around.
        pr = _st2.GetPrimAtPath(f"/World/envs/env_0/SceneObjects/{name}")
        lo = hi = None
        for c in _U2.PrimRange(pr):
            m = _UG2.Mesh(c)
            if not m:
                continue
            x = _UG2.Xformable(c).ComputeLocalToWorldTransform(_U2.TimeCode.Default())
            for v in (m.GetPointsAttr().Get() or []):
                z = x.Transform(_Gf2.Vec3d(v))[2]
                lo = z if lo is None or z < lo else lo
                hi = z if hi is None or z > hi else hi
        rz = float(objs[name].data.root_state_w[0, 2].cpu())
        return lo - rz, hi - rz

    BOTTOM_OFF = {o: _z_off(o)[0] for o, _ in ORDER}
    TOP_OFF = {b: _z_off(b)[1] for _, b in ORDER}
    print("[vec] object bottom offsets %s | bin rim offsets %s"
          % ({k: round(v, 4) for k, v in BOTTOM_OFF.items()},
             {k: round(v, 4) for k, v in TOP_OFF.items()}), flush=True)

    with open(a.dr_file) as f:
        dr_list = json.load(f)["episodes"]
    os.makedirs(a.out, exist_ok=True)

    def images():
        e = exo.data.output["rgb"][..., :3]
        w = wrist.data.output["rgb"][..., :3]
        if e.dtype != torch.uint8:
            e = (255 * e.clamp(0, 1)).to(torch.uint8); w = (255 * w.clamp(0, 1)).to(torch.uint8)
        return e.cpu().numpy(), w.cpu().numpy()

    vw = None
    if a.video:
        import cv2 as _cv
        _cols = int(np.ceil(np.sqrt(N))); _rows = int(np.ceil(N / _cols))
        _cv_fourcc = _cv.VideoWriter_fourcc(*"mp4v")
        vw = _cv.VideoWriter(a.video, _cv_fourcc, a.video_fps, (IMG_W * _cols, IMG_H * _rows))
        print(f"[vec] recording {_cols}x{_rows} tiled video -> {a.video}", flush=True)

    def _tile(imgs):
        import cv2 as _cv
        blank = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cells = [imgs[i] if i < N else blank for i in range(_rows * _cols)]
        out = np.vstack([np.hstack(cells[r * _cols:(r + 1) * _cols]) for r in range(_rows)])
        for i in range(N):
            r, c = divmod(i, _cols)
            _cv.putText(out, f"env{i}", (c * IMG_W + 6, r * IMG_H + 18),
                        _cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _cv.LINE_AA)
        return _cv.cvtColor(out, _cv.COLOR_RGB2BGR)

    total_kept = 0
    n_batches = (a.episodes + N - 1) // N
    for b in range(n_batches):
        if a.dev and b:
            # Hot-reload between batches: edit pick_sequence.py, save, and the next
            # batch runs the new waypoints without paying the Isaac boot again.
            import select
            print("\n[dev] save pick_sequence.py to reload, Enter to continue, q to quit",
                  flush=True)
            r, _, _ = select.select([_sys.stdin], [], [], 0.5)
            if r and _sys.stdin.readline().strip().lower() == "q":
                break
            _reload_seq()
            print(f"[dev] waypoints: {[p[0] for p in _seq.phases()]}", flush=True)
        idx = [a.dr_start + b * N + i for i in range(N)]
        reap_zombie_children()
        robot.write_joint_state_to_sim(q0.clone(), torch.zeros_like(q0))
        # Hold q0 during the settle. Without this the drives pull toward the last written target
        # (USD-authored pose on batch 0, the previous episode's retract afterwards), so v3 demos
        # start from scattered poses (std up to 0.4 rad). eval.py mirrors this.
        robot.set_joint_position_target(q0.clone()); scene.write_data_to_sim()
        for n, ob in objs.items():
            s0 = nominal[n].clone()
            for i, ep in enumerate(idx):
                off = dr_list[ep % len(dr_list)]
                if n in off:
                    s0[i, 0] += off[n][0]; s0[i, 1] += off[n][1]
            s0[:, 7:] = 0.0
            ob.write_root_state_to_sim(s0)
        scene.write_data_to_sim()
        for _ in range(90):
            sim.step(render=False)
        scene.update(1.0 / HZ)
        # Post-settle check on EVERY env: any object that is moving, or (for the seven
        # non-DR objects) sits >1 cm from where env0 has it, is already broken before the
        # arm moves. Graspables are excluded from the position test (DR moves them).
        _rel0 = torch.stack([objs[o].data.root_state_w[:, :3] for o in ALL_OBJS], 1) - _org[:, None]
        _vel0 = torch.stack([objs[o].data.root_state_w[:, 7:10] for o in ALL_OBJS], 1).norm(dim=-1)
        _dev = (_rel0 - _rel0[:1]).norm(dim=-1); _dev[:, :2] = 0.0
        _badm = ((_dev > 0.01) | (_vel0 > 0.1)).cpu().numpy()
        _rel0 = _rel0.cpu().numpy(); _vel0 = _vel0.cpu().numpy()
        for _e in range(N):
            for _k, _o in enumerate(ALL_OBJS):
                if _badm[_e, _k]:
                    print(f"[settle-bad] env{_e} ep{idx[_e]} {_o}: p=({_rel0[_e,_k,0]:+.3f},{_rel0[_e,_k,1]:+.3f},"
                          f"{_rel0[_e,_k,2]:+.3f}) |v|={_vel0[_e,_k]:.3f}  env0 p=({_rel0[0,_k,0]:+.3f},"
                          f"{_rel0[0,_k,1]:+.3f},{_rel0[0,_k,2]:+.3f})", flush=True)
        # Report every scene object's height after the settle, per env. Anything far
        # below the tabletop (z ~ 0) has fallen off and would poison the demo.
        import isaacsim.core.utils.stage as _su
        from pxr import Usd as _U, UsdGeom as _UG
        _st = _su.get_current_stage(); _xc = _UG.XformCache(_U.TimeCode.Default())
        for _e in range(min(N, 2)):
            _row = []
            for _n in ("mustard_bottle","spam_can","cheezit_box","soup_can","sugar_box",
                       "container","small_box","grey_bin_left","grey_bin_right","table"):
                _pr = _st.GetPrimAtPath(f"/World/envs/env_{_e}/SceneObjects/{_n}")
                if _pr.IsValid():
                    _row.append(f"{_n}={_xc.GetLocalToWorldTransform(_pr).ExtractTranslation()[2]:+.3f}")
            print(f"[fall] env{_e} " + " ".join(_row), flush=True)
        sim.render(); exo.update(0.0); wrist.update(0.0)
        for r in rmps:
            r.reset()

        traj = [[] for _ in range(N)]
        _hist.clear(); _diverged = set()
        phase_used, phase_cap = {}, {}
        for obj, binn in ORDER:
            lp, lq = grasps[obj]
            ost = objs[obj].data.root_state_w.cpu().numpy()
            bst = objs[binn].data.root_state_w.cpu().numpy()
            goals, quats, overs = [], [], []
            for i in range(N):
                g, gq = world_grasp(lp, lq, ost[i, :3], ost[i, 3:7])
                goals.append(g); quats.append(gq)
                grip_to_bottom = g[2] - (ost[i, 2] + BOTTOM_OFF[obj])
                rim_z = bst[i, 2] + TOP_OFF[binn]
                overs.append(np.array([bst[i, 0], bst[i, 1], rim_z + grip_to_bottom + 0.03]))
            for label, target_fn, grip_name, budget in _seq.phases():
                grip = CLOSE if grip_name == "close" else OPEN
                for i in range(N):
                    g, gq, ov = goals[i], quats[i], overs[i]
                    ap = q2m(gq)[:, 2]
                    rmps[i].set_goal(target_fn(g, gq, ov, ap), gq)
                # Budget is a CAP, not a fixed length. Exit as soon as EVERY env has
                # stopped moving: a fixed budget pads each waypoint with frames where the
                # arm already arrived and is standing still -- measured 1220 steps/episode
                # vs 476 for the convergence-gated sequential version. Those dead frames
                # cost storage and teach the policy to idle. Batch-wide (not per-env) so
                # the lockstep stays aligned.
                prev_hand, stable, used = None, 0, 0
                for _ in range(budget):
                    used += 1
                    q = robot.data.joint_pos.cpu().numpy()
                    qd = robot.data.joint_vel.cpu().numpy()
                    tq = np.stack([rmps[i].compute(q[i, arm].astype(float),
                                                   qd[i, arm].astype(float), 1.0 / HZ) for i in range(N)])
                    ei, wi = images()
                    if vw is not None:
                        vw.write(_tile(ei))
                    gp = np.clip(q[:, fid:fid+1] / CLOSE, 0, 1).astype(np.float32)
                    act = np.concatenate([tq, np.full((N, 1), 1.0 if grip > 0 else 0.0)], 1).astype(np.float32)
                    for i in range(N):
                        traj[i].append({"exterior_image": ei[i], "wrist_image": wi[i],
                                        "joint_position": q[i, arm].astype(np.float32),
                                        "gripper_position": gp[i], "actions": act[i]})
                    val = torch.tensor(np.concatenate([tq, np.full((N, 1), grip)], 1),
                                       dtype=torch.float32, device=robot.data.joint_pos.device)
                    robot.set_joint_position_target(val, joint_ids=arm + [fid])
                    scene.write_data_to_sim()
                    for _ in range(DECIM):
                        sim.step(render=False)
                    scene.update(1.0 / HZ)
                    sim.render(); exo.update(0.0); wrist.update(0.0)
                    # Divergence watchdog. The ~1/16 blow-up is bit-reproducible per
                    # config (identical distances across runs), so catch the FIRST step
                    # anything leaves its env and dump the last few steps: that names
                    # the phase and shows jump vs runaway.
                    _rel = torch.stack([objs[o].data.root_state_w[:, :3] for o in ALL_OBJS], 1) - _org[:, None]
                    _vel = torch.stack([objs[o].data.root_state_w[:, 7:10] for o in ALL_OBJS], 1)
                    _hnd = robot.data.body_pose_w[:, hand_idx, :3] - _org
                    _hist.append((label, used, _rel.cpu().numpy(), _vel.cpu().numpy(),
                                  _hnd.cpu().numpy(), robot.data.joint_pos.cpu().numpy()))
                    _bad = ((~torch.isfinite(_rel).all(-1)) | (_rel.norm(dim=-1) > 2.0)).any(-1)
                    _bad |= (~torch.isfinite(_hnd).all(-1)) | (_hnd.norm(dim=-1) > 3.0)
                    for _i in np.nonzero(_bad.cpu().numpy())[0]:
                        _i = int(_i)
                        if _i in _diverged:
                            continue
                        _diverged.add(_i)
                        print(f"[diverge] env{_i} ep{idx[_i]} carrying={obj} phase={label} step={used} "
                              f"global={len(traj[_i])}", flush=True)
                        for hl, hu, hr, hv, hh, hq in _hist:
                            print(f"   {hl}@{hu}: hand=({hh[_i,0]:+.3f},{hh[_i,1]:+.3f},{hh[_i,2]:+.3f}) "
                                  f"finger={hq[_i,fid]:+.3f} q={np.round(hq[_i,arm],2).tolist()}", flush=True)
                            for k, o in enumerate(ALL_OBJS):
                                print(f"      {o:14s} p=({hr[_i,k,0]:+.3f},{hr[_i,k,1]:+.3f},{hr[_i,k,2]:+.3f}) "
                                      f"|v|={float(np.linalg.norm(hv[_i,k])):.3f}", flush=True)
                    if label not in _seq.DWELLS:      # dwells must run in full
                        cur = robot.data.body_pose_w[:, hand_idx, :3].cpu().numpy()
                        # Diverged envs never settle (their hand teleports every step) and
                        # would hold the WHOLE batch at full budget: 987-1545 step demos of
                        # idling. Judge convergence on the healthy envs only.
                        _ok = np.array([i not in _diverged for i in range(N)])
                        if prev_hand is not None and (not _ok.any() or np.abs(cur - prev_hand)[_ok].max() < 0.0008):
                            stable += 1
                            if stable >= 4:
                                break
                        else:
                            stable = 0
                        prev_hand = cur
                phase_used[label] = max(phase_used.get(label, 0), used)
                phase_cap[label] = budget

        if b == 0:
            print("[phase] steps used / budget: "
                  + "  ".join(f"{k}={phase_used.get(k,0)}/{phase_cap.get(k,0)}"
                              for k, _, _, _ in _seq.phases()), flush=True)
        kept = 0
        fails = {o: 0 for o, _ in ORDER}
        for i in range(N):
            ok = True
            for obj, binn in ORDER:
                po = objs[obj].data.root_state_w[i, :3].cpu().numpy()
                pb = objs[binn].data.root_state_w[i, :3].cpu().numpy()
                good, why = in_bin(po, pb, binn)
                if not good:
                    fails[obj] += 1
                    print(f"[miss] env{i} ep{idx[i]} {obj}: {why} obj_z={po[2]:+.3f}", flush=True)
                ok &= good
            if ok:
                np.savez_compressed(os.path.join(a.out, f"ep{idx[i]:04d}.npz"),
                                    **{k: np.stack([t[k] for t in traj[i]]) for k in traj[i][0]},
                                    prompt=a.prompt)
                kept += 1
        total_kept += kept
        print(f"[vec] batch {b+1}/{n_batches}: kept {kept}/{N}  (total {total_kept})  "
              f"failures by object: {fails}", flush=True)

    if vw is not None:
        vw.release()
        # OpenCV can only write mp4v here (no libx264 in this build), which Chrome --
        # the default video/mp4 handler on this box -- cannot decode. Re-encode to real
        # H.264 with imageio-ffmpeg's bundled binary so the file plays anywhere.
        try:
            import subprocess, imageio_ffmpeg
            tmp = a.video + ".raw.mp4"
            os.replace(a.video, tmp)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                            "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "23", "-preset", "medium", a.video], check=True)
            os.remove(tmp)
        except Exception as e:
            print(f"[vec] H.264 re-encode skipped ({e})", flush=True)
        print(f"[vec] video -> {a.video}", flush=True)
    print(f"\n[vec] KEPT {total_kept}/{n_batches * N} -> {a.out}", flush=True)


main()
# app.close() hangs on this box, so a finished run used to linger for its full timeout,
# holding ~5 GB of GPU and two zombie children (Omniverse Hub, nvidia-ngx-updater --
# helpers Kit never reaps; they vanish with the parent). Everything is already on disk
# (savez writes synchronously), so skip the teardown and leave the process immediately.
import sys as _sys2, os as _os
_sys2.stdout.flush(); _sys2.stderr.flush()
_os._exit(0)
