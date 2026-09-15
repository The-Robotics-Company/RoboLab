# SPDX-License-Identifier: Apache-2.0
"""Vectorised pi05 eval on food_packing: N envs in one Isaac, lockstep, one policy server.

Per episode: DR file draw, demo start poses, latched in_bin success,
early stop, diagnostics, results.json) but 20 episodes run as one batch: the sim steps all
envs together and the server answers N requests per 15-step chunk. The old single-env eval took ~35 min / 20 episodes; this takes ~4 min.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_physics import apply_scene_physics, add_physics_args, in_bin, reap_zombie_children, select_scene, ALL_SCENE_OBJS  # noqa: E402
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--num-envs", type=int, default=20)
p.add_argument("--env-spacing", type=float, default=14.0)
p.add_argument("--max-steps", type=int, default=600)
p.add_argument("--host", default="localhost")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--prompt", default="put the mustard bottle in the left bin and the spam can in the right bin")
p.add_argument("--out", default="/tmp/pi05_eval_vec")
p.add_argument("--dr-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dr_episodes.json"))
p.add_argument("--dr-start", type=int, default=0)
p.add_argument("--dr-list", default="", help="comma-separated DR indices instead of dr-start..+episodes")
p.add_argument("--home-from-demos", default="",
               help="episode i starts from demo i's first joint pose ('' = config default pose)")
p.add_argument("--obs-format", default="droid", choices=["droid", "robolab"])
p.add_argument("--dry-run", action="store_true", help="no server: hold the start pose")
p.add_argument("--video", default="", help="tiled mp4 of the exterior camera, all envs")
p.add_argument("--video-wrist", default="", help="tiled mp4 of the wrist camera, all envs")
add_physics_args(p)
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.enable_cameras = True
select_scene(a.scene)
app = AppLauncher(a).app

import numpy as np, torch                                          # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext          # noqa: E402
from scene_cfg import build_scene, IMG_W, IMG_H                    # noqa: E402

HZ, DECIM = 15, 8
TARGETS = {"mustard_bottle": "grey_bin_left", "spam_can": "grey_bin_right"}
CLOSE = float(np.pi / 4)


def main():
    t0 = time.time()
    ep_ids = ([int(x) for x in a.dr_list.split(",")] if a.dr_list
              else [a.dr_start + i for i in range(a.episodes)])
    N = min(a.num_envs, len(ep_ids))
    sim = SimulationContext(SimulationCfg(dt=1.0 / (HZ * DECIM), device=a.device))
    scene = build_scene(N, a.env_spacing, a.wrist_cam, IMG_W, IMG_H, a.scene)
    import isaacsim.core.utils.stage as stage_utils                 # noqa: E402
    phys = apply_scene_physics(stage_utils.get_current_stage(), pin_bins=a.pin_bins,
                               pin_distractors=a.pin_distractors, bin_collider=a.bin_collider,
                               contact_tune=a.contact_tune, depen_cap=a.depen_cap,
                               loop_closure=a.loop_closure, distractor_tune=a.distractor_tune,
                               obj_vel_cap=a.obj_vel_cap, distractor_collider=a.distractor_collider,
                               graspable_collider=a.graspable_collider, set_mass=a.set_mass)
    print(f"[vec-eval] {N} envs | scene {a.scene} | wrist cam {a.wrist_cam} | {phys}", flush=True)
    sim.reset()
    for _ in range(a.pre_settle):
        sim.step(render=False)
    if a.pre_settle:
        scene.update(1.0 / HZ)
    robot, exo, wrist = scene["robot"], scene["exo"], scene["wrist"]
    objs = {n: scene[n] for n in ALL_SCENE_OBJS}
    nominal = {n: objs[n].data.root_state_w.clone() for n in objs}
    jn = list(robot.data.joint_names)
    arm = [jn.index(f"panda_joint{i}") for i in range(1, 8)]
    fid = jn.index("finger_joint")
    q0 = robot.data.default_joint_pos.clone()
    dev = q0.device

    with open(a.dr_file) as f:
        dr_list = json.load(f)["episodes"]
    homes = []
    if a.home_from_demos:
        import glob
        for fpath in sorted(glob.glob(os.path.join(a.home_from_demos, "ep*.npz"))):
            with np.load(fpath) as z:
                homes.append(z["joint_position"][0].astype(np.float32))
        print(f"[vec-eval] start poses from {len(homes)} demos", flush=True)

    client = None
    if not a.dry_run:
        from openpi_client import image_tools, websocket_client_policy
        client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
        print(f"[vec-eval] connected to {a.host}:{a.port}", flush=True)
    else:
        from openpi_client import image_tools
        print("[vec-eval] DRY RUN: holding start pose", flush=True)
    k_ext, k_wr = (("observation/exterior_image_1_left", "observation/wrist_image_left")
                   if a.obs_format == "droid" else ("observation/image", "observation/wrist_image"))
    os.makedirs(a.out, exist_ok=True)

    vw = None
    if a.video:
        import cv2
        cols = int(np.ceil(np.sqrt(N))); rows = int(np.ceil(N / cols))
        vw = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), HZ, (IMG_W * cols, IMG_H * rows))
    vww = None
    if a.video_wrist:
        import cv2
        cols = int(np.ceil(np.sqrt(N))); rows = int(np.ceil(N / cols))
        vww = cv2.VideoWriter(a.video_wrist, cv2.VideoWriter_fourcc(*"mp4v"), HZ, (IMG_W * cols, IMG_H * rows))

    def images():
        e = exo.data.output["rgb"][..., :3]; w = wrist.data.output["rgb"][..., :3]
        if e.dtype != torch.uint8:
            e = (255 * e.clamp(0, 1)).to(torch.uint8); w = (255 * w.clamp(0, 1)).to(torch.uint8)
        return e.cpu().numpy(), w.cpu().numpy()

    def tile(imgs, labels):
        import cv2
        blank = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cells = [imgs[i] if i < N else blank for i in range(rows * cols)]
        out = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
        for i in range(N):
            r, c = divmod(i, cols)
            cv2.putText(out, labels[i], (c * IMG_W + 6, r * IMG_H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    results = []
    n_batches = (len(ep_ids) + N - 1) // N
    for b in range(n_batches):
        reap_zombie_children()
        idx = ep_ids[b * N:(b + 1) * N]
        active = list(range(len(idx)))          # envs with a real episode this batch
        # ---- reset: robot start pose (held during settle), all objects nominal + DR
        q_home = q0.clone()
        for i, dr_i in enumerate(idx):
            if homes:
                q_home[i, arm] = torch.tensor(homes[dr_i % len(homes)], dtype=q_home.dtype, device=dev)
        robot.write_joint_state_to_sim(q_home, torch.zeros_like(q_home))
        robot.set_joint_position_target(q_home)
        for n, ob in objs.items():
            s0 = nominal[n].clone()
            for i, dr_i in enumerate(idx):
                off = dr_list[dr_i % len(dr_list)]
                if n in off:
                    s0[i, 0] += off[n][0]; s0[i, 1] += off[n][1]
            s0[:, 7:] = 0.0
            ob.write_root_state_to_sim(s0)
        scene.write_data_to_sim()
        for _ in range(90):
            sim.step(render=False)
        scene.update(1.0 / HZ); sim.render(); exo.update(0.0); wrist.update(0.0)
        spawn = {o: objs[o].data.root_state_w[:, :3].cpu().numpy().copy() for o in TARGETS}

        latched = {o: np.zeros(N, bool) for o in TARGETS}
        done = np.zeros(N, bool); blown = np.zeros(N, bool); solved_at = np.full(N, -1)
        chunk = np.zeros((N, 15, 8), np.float32); traj_jp, traj_act = [], []
        jp0 = robot.data.joint_pos[:, arm].cpu().numpy().copy()
        t_inf = 0.0
        for step in range(a.max_steps):
            q = robot.data.joint_pos.cpu().numpy()
            jp = q[:, arm].astype(np.float32)
            gp = np.clip(q[:, fid:fid + 1] / CLOSE, 0, 1).astype(np.float32)
            ei, wi = images()
            _labels = [f"ep{idx[i]} {'OK' if done[i] and not blown[i] else ('X' if blown[i] else '')}" if i < len(idx) else "" for i in range(N)]
            if vw is not None:
                vw.write(tile(ei, _labels))
            if vww is not None:
                vww.write(tile(wi, _labels))
            if step % 15 == 0:
                if a.dry_run:
                    chunk[:] = np.concatenate([q_home[:, arm].cpu().numpy(), np.zeros((N, 1))], 1)[:, None, :]
                else:
                    t1 = time.time()
                    for i in active:
                        if done[i]:
                            continue
                        req = {k_ext: image_tools.resize_with_pad(ei[i], 224, 224),
                               k_wr: image_tools.resize_with_pad(wi[i], 224, 224),
                               "observation/joint_position": jp[i],
                               "observation/gripper_position": gp[i],
                               "prompt": a.prompt}
                        chunk[i] = np.asarray(client.infer(req)["actions"])[:15, :8]
                    t_inf += time.time() - t1
            act = chunk[:, step % 15]
            traj_jp.append(jp); traj_act.append(act.copy())
            tgt = robot.data.joint_pos.clone()
            act_t = torch.tensor(act, dtype=tgt.dtype, device=dev)
            live = torch.tensor(~done, device=dev)
            tgt[:, arm] = torch.where(live[:, None], act_t[:, :7], tgt[:, arm])
            tgt[:, fid] = torch.where(live, (act_t[:, 7] > 0.5).to(tgt.dtype) * CLOSE, tgt[:, fid])
            robot.set_joint_position_target(tgt)
            scene.write_data_to_sim()
            for _ in range(DECIM):
                sim.step(render=False)
            scene.update(1.0 / HZ); sim.render(); exo.update(0.0); wrist.update(0.0)
            # success latch + blow-up detection, per env
            qn = robot.data.joint_pos.cpu().numpy()
            newly_blown = (~np.isfinite(qn).all(1)) | (np.abs(qn).max(1) > 100)
            blown |= newly_blown
            for o, bn in TARGETS.items():
                po = objs[o].data.root_state_w[:, :3].cpu().numpy(); pb = objs[bn].data.root_state_w[:, :3].cpu().numpy()
                now = np.array([in_bin(po[i], pb[i], bn)[0] for i in range(N)])
                latched[o] |= now
            solved = latched["mustard_bottle"] & latched["spam_can"] & ~done & ~blown
            solved_at[solved] = step
            done |= solved | blown
            if all(done[i] for i in active):
                break
        steps = step + 1
        jp_all = np.stack(traj_jp); act_all = np.stack(traj_act)
        for i, dr_i in enumerate(idx):
            final = {o: in_bin(objs[o].data.root_state_w[i, :3].cpu().numpy(), objs[bn].data.root_state_w[i, :3].cpu().numpy(), bn)[0]
                     for o, bn in TARGETS.items()}
            ok = {o: bool((latched[o][i] or final[o]) and not blown[i]) for o in TARGETS}
            succ = all(ok.values())
            exc = float(np.abs(jp_all[:, i] - jp0[i]).max())
            r = {"episode": dr_i, "dr_index": dr_i, "success": succ, "solved_at_step": int(solved_at[i]) if solved_at[i] >= 0 else None,
                 **{f"in_{k}": v for k, v in ok.items()}, "blowup": bool(blown[i]),
                 "dr": {k: [float(v[0]), float(v[1])] for k, v in dr_list[dr_i % len(dr_list)].items()},
                 "steps": int(steps), "max_joint_excursion_rad": exc,
                 "frac_steps_gripper_closed_cmd": float((act_all[:, i, 7] > 0.5).mean()),
                 "obj_disp_m": {o: float(np.linalg.norm(objs[o].data.root_state_w[i, :3].cpu().numpy() - spawn[o][i])) for o in TARGETS}}
            results.append(r)
            print(f"[vec-eval] dr {dr_i}: success={succ} {ok}{' BLOWUP' if blown[i] else ''} | excursion={exc:.2f} "
                  f"closed_cmd={r['frac_steps_gripper_closed_cmd']:.2f} disp={ {k: round(v, 3) for k, v in r['obj_disp_m'].items()} }", flush=True)
        print(f"[vec-eval] batch {b + 1}/{n_batches}: {sum(r['success'] for r in results[-len(idx):])}/{len(idx)} in {steps} steps, "
              f"inference {t_inf:.0f}s, elapsed {time.time() - t0:.0f}s", flush=True)

    for _w, _path in ((vw, a.video), (vww, a.video_wrist)):
        if _w is None:
            continue
        _w.release()
        try:
            import subprocess, imageio_ffmpeg
            tmp = _path + ".raw.mp4"; os.replace(_path, tmp)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "23", _path], check=True)
            os.remove(tmp)
        except Exception as e:
            print(f"[vec-eval] H.264 re-encode skipped ({e})", flush=True)
        print(f"[vec-eval] video -> {_path}", flush=True)
    n = len(results); s = sum(r["success"] for r in results)
    summary = {"episodes": n, "successes": s, "success_rate": s / n if n else 0.0,
               "per_object": {k: sum(r[f"in_{k}"] for r in results) / n for k in TARGETS},
               "blowups": sum(r["blowup"] for r in results), "prompt": a.prompt, "obs_format": a.obs_format,
               "wrist_cam": a.wrist_cam, "scene": a.scene, "results": results}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[vec-eval] SUCCESS RATE {s}/{n} = {100 * s / max(n, 1):.1f}%  per-object {summary['per_object']}  "
          f"blowups {summary['blowups']}  total {time.time() - t0:.0f}s", flush=True)


main()
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
