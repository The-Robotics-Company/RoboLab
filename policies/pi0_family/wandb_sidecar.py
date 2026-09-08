# SPDX-License-Identifier: Apache-2.0

"""Tail a RoboLab evaluation output folder and log metrics + episode videos to W&B.

Runs next to run_rollout.py (or any run_evaluation-based runner): follows episode_results.jsonl, logs one
W&B step per finished episode (success, score, steps, duration, running success rate) with the policy-view
and viewport mp4s, then a per-episode table and final summary when the runner exits.

Usage:
  python wandb_sidecar.py --output-dir <repo>/output/<label> --runner-pid <pid> --run-name <name> \
      --project piperx-robolab --num-envs 5 [--config-json '{"robot": "piperx", ...}']
"""

import argparse
import collections
import json
import os
import re
import subprocess
import time

import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", required=True)
ap.add_argument("--runner-pid", type=int, required=True)
ap.add_argument("--run-name", required=True)
ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "piperx-robolab"))
ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
ap.add_argument("--num-envs", type=int, default=1, help="parallel envs per run (maps episode idx -> run/env video names)")
ap.add_argument("--config-json", default="{}", help="extra config fields for the W&B run")
ap.add_argument("--poll-s", type=float, default=15.0)
args = ap.parse_args()

results_path = os.path.join(args.output_dir, "episode_results.jsonl")
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "?"


config = {"robolab_commit": _git("rev-parse", "--short", "HEAD"), "host": os.uname().nodename,
          "output_dir": args.output_dir, "num_envs": args.num_envs}
config.update(json.loads(args.config_json))
run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, config=config)
run.define_metric("*", step_metric="episode_idx")


def video_paths(r):
    task, inst = r["env_name"], r.get("instruction") or ""
    cleaned = re.sub(r"[^\w\s]", "", inst).replace(" ", "_")
    run_idx = int(r.get("run", int(r.get("episode", 0)) // args.num_envs))
    env_id = int(r.get("env_id", int(r.get("episode", 0)) % args.num_envs))
    base = os.path.join(args.output_dir, task, f"{cleaned}_{run_idx}_env{env_id}")
    sensor, view = base + ".mp4", base + "_viewport.mp4"
    return (sensor if os.path.exists(sensor) else None), (view if os.path.exists(view) else None)


def wait_stable(path, tries=30):
    """Wait until the mp4 stops growing (writer closed)."""
    if not path:
        return None
    last = -1
    for _ in range(tries):
        size = os.path.getsize(path)
        if size == last and size > 0:
            return path
        last = size
        time.sleep(1)
    return path


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


seen = 0
per_task = collections.defaultdict(list)
t0 = time.time()
table = wandb.Table(columns=["task", "run", "env_id", "success", "score", "steps", "duration_s", "reason",
                             "policy_view", "viewport"])


def flush():
    global seen
    if not os.path.exists(results_path):
        return
    with open(results_path) as f:
        lines = f.readlines()
    for line in lines[seen:]:
        line = line.strip()
        if not line:
            seen += 1
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            return  # partial line, retry next poll
        seen += 1
        task = r.get("env_name")
        succ = int(bool(r.get("success")))
        per_task[task].append(succ)
        all_eps = [s for v in per_task.values() for s in v]
        log = {"episode_idx": len(all_eps), "task": task, "run": r.get("run"), "env_id": r.get("env_id"),
               "success": succ, "score": r.get("score"), "steps": r.get("episode_step"),
               "duration_s": r.get("duration"), "agg/success_rate": sum(all_eps) / len(all_eps),
               "agg/n_episodes": len(all_eps), "agg/wall_min": (time.time() - t0) / 60,
               f"task/{task}/success_rate": sum(per_task[task]) / len(per_task[task])}
        sensor, view = video_paths(r)
        cap = f"{task} run{r.get('run')} env{r.get('env_id')} {'SUCCESS' if succ else 'FAIL'} score={r.get('score')} | {r.get('instruction', '')}"
        if wait_stable(sensor):
            log["videos/policy_view"] = wandb.Video(sensor, caption=cap, format="mp4")
        if wait_stable(view):
            log["videos/viewport"] = wandb.Video(view, caption=cap, format="mp4")
        table.add_data(task, r.get("run"), r.get("env_id"), succ, r.get("score"), r.get("episode_step"),
                       r.get("duration"), str(r.get("reason")),
                       wandb.Video(sensor, format="mp4") if sensor else None,
                       wandb.Video(view, format="mp4") if view else None)
        run.log(log)
        print(f"[sidecar] {task} run{r.get('run')} env{r.get('env_id')} success={succ} "
              f"videos={bool(sensor)}/{bool(view)}", flush=True)


while True:
    flush()
    if not alive(args.runner_pid):
        time.sleep(5)
        flush()
        break
    time.sleep(args.poll_s)

all_eps = [s for v in per_task.values() for s in v]
summary = {t: {"success_rate": sum(v) / len(v), "n": len(v)} for t, v in per_task.items()}
run.log({"episodes": table})
run.summary["final/success_rate"] = sum(all_eps) / max(1, len(all_eps))
run.summary["final/n_episodes"] = len(all_eps)
run.summary["final/per_task"] = json.dumps(summary)
for extra in ("episode_results.jsonl",):
    p = os.path.join(args.output_dir, extra)
    if os.path.exists(p):
        run.save(p, base_path=args.output_dir, policy="now")
run.finish()
print("[sidecar] done", json.dumps(summary), flush=True)
