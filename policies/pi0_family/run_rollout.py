# SPDX-License-Identifier: Apache-2.0

"""Roll out a pi0-family policy in RoboLab: robot + task(s) + scene variant + checkpoint -> episodes.

Same evaluation loop as ``run.py`` (robolab.eval.runner.run_evaluation): HDF5 + videos + summary under
<repo>/output/. Additions:
  --robot piperx|droid    piperx: spawn the PiPER-X (robolab.robots.piper_x, exo + wrist DC1 cameras, 7-dim
                          actions) with Pi0PiperXClient. droid: RoboLab's stock Franka + Robotiq DROID setup
                          (over-shoulder + wrist cameras, 8-dim actions) with the shipped Pi0DroidJointposClient.
  --scene-variant gt|v0   gt uses each task's RoboLab scene; v0 swaps in the scene file with the same name plus
                          a ``_v0`` suffix (rubiks_cube_bowl.usda -> rubiks_cube_bowl_v0.usda, anywhere under
                          assets/scenes). Applied before task registration, so success terms, instructions and
                          contact objects come from the unchanged task class.
  --checkpoint DIR        openpi checkpoint (local path or gs://). Starts an openpi policy server for the run
                          (``scripts/serve_policy.py --port N policy:checkpoint ...``) and stops it afterwards.
                          Without it, connects to an already running server on --remote-host/--remote-port.
  --policy-config NAME    openpi training-config name of the checkpoint (default for droid: pi05_droid_jointpos;
                          required for piperx when --checkpoint is given).
  --action-space abs|delta  piperx only: absolute joint targets or per-step deltas (droid is joint-position).
  --rig NAME|PATH|stock   Scene rig USD spawned once at /World for either robot: the HDR dome (light + backdrop)
                          and the visible ground, from stages/rigs/<NAME>.usda (default home_office, the same
                          file the inspection stages reference). stock: RoboLab's per-robot lighting/background
                          cfgs instead (droid: SphereLight + --background HDR).
  --rig-ground auto|Z     where the rig's visible ground sits: auto (default) = the task scene's own /GroundPlane z.

Examples:
  .venv/bin/python policies/pi0_family/run_rollout.py --headless --robot piperx --task RubiksCubeTask \
      --scene-variant v0 --checkpoint /path/to/ckpt --policy-config pi05_piperx --num-envs 4
  .venv/bin/python policies/pi0_family/run_rollout.py --headless --robot droid --task RubiksCubeTask \
      --checkpoint gs://openpi-assets-simeval/pi05_droid_jointpos
  .venv/bin/python policies/pi0_family/run_rollout.py --headless --robot piperx --task RubiksCubeTask   # server already up on :8000
"""

import argparse
import atexit
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

PI0_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]
ROBOTS = ["piperx", "droid"]
DEFAULT_POLICY_CONFIG = {"droid": "pi05_droid_jointpos", "piperx": None}

parser = argparse.ArgumentParser(description="Roll out a Pi0-family policy in RoboLab (PiPER-X or Franka DROID).")
parser.add_argument("--robot", choices=ROBOTS, default="piperx",
                    help="Robot to spawn in the task scene (default: piperx).")
parser.add_argument("--policy", choices=PI0_VARIANTS, default="pi05",
                    help="Pi0-family variant (default: pi05). Selects the client's default open-loop horizon.")
parser.add_argument("--scene-variant", "--scene_variant", choices=["gt", "v0"], default="gt",
                    help="gt: the task's RoboLab scene. v0: same scene name with a _v0 suffix (default: gt).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="openpi checkpoint dir (local or gs://). Starts a policy server for this run.")
parser.add_argument("--policy-config", "--policy_config", type=str, default=None,
                    help="openpi config name the checkpoint was trained with "
                         "(default: pi05_droid_jointpos for --robot droid; required for piperx with --checkpoint).")
parser.add_argument("--openpi-dir", "--openpi_dir", type=str, default=os.path.expanduser("~/git/trc-policy-lab/openpi"),
                    help="openpi checkout used to start the server (default: ~/git/trc-policy-lab/openpi).")
parser.add_argument("--server-startup-timeout", type=float, default=1800.0,
                    help="Seconds to wait for the policy server port (default: 1800; big checkpoints load slowly).")
parser.add_argument("--server-mem-fraction", type=str, default="0.5",
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION for a spawned openpi server (default 0.5 of the GPU).")
parser.add_argument("--warmup-timeout", type=float, default=900.0,
                    help="Seconds to keep retrying a dummy inference until the server has JIT-compiled (default 900). "
                         "0 disables the warm-up.")
parser.add_argument("--action-space", "--action_space", choices=["abs", "delta"], default="abs",
                    help="piperx only: absolute joint targets or per-step deltas (default: abs).")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000)
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                    help="Full WebSocket URI for the policy server; overrides host/port.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help="Actions executed per predicted chunk (default: per-variant; must match action_horizon).")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                    help="Also record camera images into the HDF5 (default: proprio only).")
parser.add_argument("--rig", type=str, default="home_office",
                    help="Scene rig USD (stages/rigs/<name>.usda, or a path) spawned once at /World: HDR dome as light + "
                         "backdrop and the visible ground, identical to the inspection stages. Default home_office. "
                         "'stock' = RoboLab's per-robot lighting/background cfgs instead.")
parser.add_argument("--background", type=str, default="home_office",
                    help="--rig stock only: RoboLab HDR background cfg: home_office, empty_warehouse, billiard_hall, "
                         "brown_photostudio, or none.")
parser.add_argument("--rig-ground", "--rig_ground", type=str, default="auto",
                    help="Height of the rig's visible ground plane: 'auto' (default) reads the task scene's authored "
                         "/GroundPlane z, so the visible ground sits on the scene's collider ground; or give a z in metres.")
parser.add_argument("--droid-compat", "--droid_compat", action="store_true",
                    help="piperx only: drive the 6-joint Piper-X with a DROID-trained (7-joint, 8-dim) checkpoint by "
                         "padding the joint state and dropping the 7th joint action. Auto-enabled when --policy-config "
                         "contains 'droid'. Pipeline test only, not a meaningful transfer.")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true",
                    help="droid only: sample a random non-default background per task at registration time.")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None,
                    help="droid only: seed for --randomize-background.")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if args_cli.policy_config is None:
    args_cli.policy_config = DEFAULT_POLICY_CONFIG[args_cli.robot]
if args_cli.checkpoint is not None and args_cli.policy_config is None:
    parser.error("--policy-config is required with --checkpoint for --robot piperx "
                 "(no default openpi config exists for the PiPER-X yet).")
if args_cli.robot == "piperx" and not args_cli.droid_compat and args_cli.policy_config and "droid" in args_cli.policy_config:
    args_cli.droid_compat = True
    print("\033[93m[RoboLab] --policy-config looks DROID-trained: enabling --droid-compat for the Piper-X client.\033[0m")
if args_cli.robot == "droid" and args_cli.action_space != "abs":
    print("\033[93m[RoboLab] --action-space is ignored for --robot droid (joint-position registration).\033[0m")
if args_cli.robot == "piperx" and (args_cli.randomize_background or args_cli.background_seed is not None):
    print("\033[93m[RoboLab] --randomize-background/--background-seed are ignored for --robot piperx.\033[0m")


# ----------------------------------------------------------------------------- policy server (optional)
def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def start_policy_server(args: argparse.Namespace) -> subprocess.Popen | None:
    """Launch openpi's serve_policy.py for --checkpoint, or return None if no checkpoint was given."""
    if args.checkpoint is None:
        return None
    if _port_open(args.remote_host, args.remote_port):
        # A previous run's server may have outlived its runner (SimulationApp.close() can hard-exit the process
        # before the finally/atexit hooks run). If the port holder is an openpi serve_policy.py, replace it.
        stale = subprocess.run(["pgrep", "-f", "scripts/serve_policy.py"], capture_output=True, text=True).stdout.split()
        if not stale:
            raise RuntimeError(
                f"Port {args.remote_port} is already in use by something other than an openpi server. Either drop "
                "--checkpoint to use the running server, or pick another --remote-port."
            )
        print(f"\033[93m[RoboLab] Port {args.remote_port} held by a stale openpi server (pid {', '.join(stale)}); "
              f"stopping it before starting the new one.\033[0m")
        subprocess.run(["pkill", "-f", "scripts/serve_policy.py"], check=False)
        for _ in range(30):
            if not _port_open(args.remote_host, args.remote_port):
                break
            time.sleep(1.0)
        else:
            subprocess.run(["pkill", "-9", "-f", "scripts/serve_policy.py"], check=False)
            time.sleep(3.0)
    python = os.path.join(args.openpi_dir, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise FileNotFoundError(f"openpi venv python not found at {python} (set --openpi-dir)")
    cmd = [
        python, "scripts/serve_policy.py", "--port", str(args.remote_port),
        "policy:checkpoint", f"--policy.config={args.policy_config}", f"--policy.dir={args.checkpoint}",
    ]
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output", "policy_server_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"serve_{args.policy_config}_{int(time.time())}.log")
    print(f"\033[96m[RoboLab] Starting policy server: {shlex.join(cmd)}\n           log: {log_path}\033[0m")
    # Cap JAX's GPU preallocation: by default it grabs ~75% of the card and leaves Isaac Sim nothing.
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", args.server_mem_fraction)
    proc = subprocess.Popen(cmd, cwd=args.openpi_dir, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    atexit.register(stop_policy_server, proc)
    t0 = time.time()
    while not _port_open(args.remote_host, args.remote_port):
        if proc.poll() is not None:
            raise RuntimeError(f"Policy server exited with code {proc.returncode}; see {log_path}")
        if time.time() - t0 > args.server_startup_timeout:
            raise TimeoutError(f"Policy server did not open port {args.remote_port} within "
                               f"{args.server_startup_timeout:.0f} s; see {log_path}")
        time.sleep(2.0)
    print(f"\033[96m[RoboLab] Policy server ready on {args.remote_host}:{args.remote_port} "
          f"after {time.time() - t0:.0f} s.\033[0m")
    return proc


def stop_policy_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


# ----------------------------------------------------------------------------- scene variant (gt | v0)
def install_scene_variant(variant: str) -> None:
    """For v0, make every scene lookup resolve ``<name>_v0.<ext>`` instead of ``<name>.<ext>``.

    Task classes call ``import_scene("<name>.usda", ...)`` at class-definition time, and RoboLab executes task
    files fresh at registration, so patching the lookup before registration retargets every selected task.
    """
    if variant == "gt":
        return
    import robolab.core.scenes.utils as scene_utils

    original = scene_utils.find_scene_file

    def find_scene_file_v0(scene_path: str, scene_dir: str, *a, **kw) -> str:
        if os.path.isabs(scene_path):
            return scene_path
        stem, ext = os.path.splitext(scene_path)
        if stem.endswith("_v0"):
            return original(scene_path, scene_dir, *a, **kw)
        v0_name = f"{stem}_v0{ext}"
        resolved = original(v0_name, scene_dir, *a, **kw)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"--scene-variant v0: no scene file '{v0_name}' under {scene_dir} for task scene '{scene_path}'."
            )
        print(f"\033[96m[RoboLab] scene variant v0: {scene_path} -> {os.path.relpath(resolved, scene_dir)}\033[0m")
        return resolved

    scene_utils.find_scene_file = find_scene_file_v0


# ----------------------------------------------------------------------------- rig ground height
def resolve_rig_ground(args: argparse.Namespace) -> float:
    """--rig-ground: an explicit z, or 'auto' = the selected task's scene ground (its authored /GroundPlane)."""
    import json

    from robolab.constants import SCENE_DIR, TASK_DIR
    from robolab.core.scenes import utils as scene_utils
    from robolab.registrations.rig import DEFAULT_GROUND_Z, scene_ground_z

    if args.rig_ground.lower() != "auto":
        return float(args.rig_ground)
    grounds = []
    try:
        meta = json.load(open(os.path.join(TASK_DIR, "_metadata", "task_metadata.json")))
        entries = meta if isinstance(meta, list) else meta.get("tasks", [])
        by_name = {e["task_name"]: e.get("scene") for e in entries if isinstance(e, dict) and "task_name" in e}
        for task in args.task or []:
            scene = by_name.get(task)
            if scene:
                grounds.append(round(scene_ground_z(scene_utils.find_scene_file(scene, SCENE_DIR)), 6))
    except Exception as exc:  # noqa: BLE001
        print(f"\033[93m[RoboLab] --rig-ground auto failed ({exc}); using {DEFAULT_GROUND_Z}.\033[0m")
    if not grounds:
        print(f"\033[93m[RoboLab] --rig-ground auto: no task scene ground found; using {DEFAULT_GROUND_Z}.\033[0m")
        return DEFAULT_GROUND_Z
    if len(set(grounds)) > 1:
        print(f"\033[93m[RoboLab] --rig-ground auto: tasks disagree on ground height {sorted(set(grounds))}; using "
              f"{grounds[0]}. Pass --rig-ground explicitly or run one task per job.\033[0m")
    return grounds[0]


# ----------------------------------------------------------------------------- robot selection
def register_robot_envs(args: argparse.Namespace) -> None:
    """Register the selected robot's envs for the selected tasks (call after AppLauncher, before run_evaluation)."""
    from robolab.registrations.piperx.auto_env_registrations_jointpos import resolve_background
    from robolab.registrations.rig import rig_cfg, rig_ground_z, rig_usd_path

    stock_rig = (not args.rig) or args.rig.lower() in ("stock", "none")
    rig = None if stock_rig else rig_cfg(args.rig, ground_z=resolve_rig_ground(args))
    if rig is not None:
        print(f"[RoboLab] scene rig: {rig_usd_path(rig)} at ground z={rig_ground_z(rig):.4f} (spawned once at /World/rig)")
    if args.robot == "piperx":
        from robolab.registrations.piperx.auto_env_registrations_jointpos import (
            auto_register_piperx_delta_envs,
            auto_register_piperx_envs,
            resolve_background,
        )
        register = auto_register_piperx_envs if args.action_space == "abs" else auto_register_piperx_delta_envs
        if rig is not None:
            register(task_dirs=args.task_dirs, task=args.task, lighting_cfg=rig, background_cfg=None)
        else:   # stock: no rig, RoboLab-style separate cfgs (piperx has no stock light, only the HDR background)
            register(task_dirs=args.task_dirs, task=args.task, lighting_cfg=None,
                     background_cfg=resolve_background(args.background))
    else:
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

        kwargs = dict(task_dirs=args.task_dirs, task=args.task, randomize_background=args.randomize_background,
                      background_seed=args.background_seed)
        if rig is not None:
            if args.randomize_background:
                parser.error("--randomize-background needs --rig stock (a rig USD carries its own HDR).")
            kwargs.update(lighting_cfg=rig, background_cfg=None)
        elif not args.randomize_background:
            kwargs.update(background_cfg=resolve_background(args.background))
        auto_register_droid_envs(**kwargs)


def warm_up_policy_server(args: argparse.Namespace) -> None:
    """Send dummy inferences until one succeeds, so JAX's first-call JIT compile (minutes) happens before any
    episode starts. During that compile the server stops answering websocket pings, the client drops the
    connection and its reconnects time out, which kills the run if it happens mid-episode."""
    if args.warmup_timeout <= 0:
        return
    import numpy as np
    import torch

    from policies.pi0_family.piperx_client import Pi0PiperXClient

    t0 = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            if args.robot == "piperx":
                client = Pi0PiperXClient(remote_host=args.remote_host, remote_port=args.remote_port,
                                         remote_uri=args.remote_uri, policy_variant=args.policy,
                                         droid_compat=args.droid_compat)
                fake = {"image_obs": {"exo_camera": [torch.zeros((352, 624, 3), dtype=torch.uint8)],
                                      "wrist_camera": [torch.zeros((352, 624, 3), dtype=torch.uint8)]},
                        "proprio_obs": {"arm_joint_pos": torch.zeros((1, 6)), "gripper_pos": torch.zeros((1, 1))}}
            else:
                from policies.pi0_family.client import Pi0DroidJointposClient

                client = Pi0DroidJointposClient(remote_host=args.remote_host, remote_port=args.remote_port,
                                                remote_uri=args.remote_uri, policy_variant=args.policy)
                fake = {"image_obs": {"over_shoulder_left_camera": [torch.zeros((224, 224, 3), dtype=torch.uint8)],
                                      "wrist_cam": [torch.zeros((224, 224, 3), dtype=torch.uint8)]},
                        "proprio_obs": {"arm_joint_pos": torch.zeros((1, 7)), "gripper_pos": torch.zeros((1, 1))}}
            t1 = time.time()
            out = client.infer(fake, "warm-up")
            action = np.asarray(out["action"])
            print(f"\033[96m[RoboLab] Policy server warm: first inference {time.time() - t1:.1f} s, action shape "
                  f"{action.shape}, {time.time() - t0:.0f} s total, {attempt} attempt(s).\033[0m")
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            return
        except Exception as e:  # noqa: BLE001  (connection drops while the server compiles)
            if time.time() - t0 > args.warmup_timeout:
                raise TimeoutError(f"policy server did not answer a warm-up inference within {args.warmup_timeout:.0f} s") from e
            print(f"\033[93m[RoboLab] warm-up attempt {attempt} failed ({type(e).__name__}); server still compiling? "
                  f"retrying in 15 s ({time.time() - t0:.0f} s elapsed)\033[0m")
            time.sleep(15.0)


def make_client(args: argparse.Namespace):
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        policy_variant=args.policy,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if args.robot == "piperx":
        from policies.pi0_family.piperx_client import Pi0PiperXClient

        return Pi0PiperXClient(droid_compat=args.droid_compat, **kwargs)
    from policies.pi0_family.client import Pi0DroidJointposClient

    return Pi0DroidJointposClient(**kwargs)


# ----------------------------------------------------------------------------- main
server_proc = start_policy_server(args_cli)   # before Isaac boots, so the checkpoint loads while Kit starts

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

warm_up_policy_server(args_cli)   # JIT-compile the policy before the first episode (see docstring)
install_scene_variant(args_cli.scene_variant)
register_robot_envs(args_cli)


def main() -> None:
    # Output folder / env_cfg.policy label carries robot, action space, scene variant and checkpoint for provenance.
    ckpt_tag = os.path.basename(args_cli.checkpoint.rstrip("/")) if args_cli.checkpoint else "server"
    space = args_cli.action_space if args_cli.robot == "piperx" else "jointpos"
    # The rig is named only when it departs from the default, so existing folder names stay stable.
    rig_name = os.path.splitext(os.path.basename(args_cli.rig))[0] if args_cli.rig else "stock"
    rig_tag = "" if rig_name == "home_office" else f"_rig-{rig_name}"
    label = f"{args_cli.policy}_{args_cli.robot}_{space}_{args_cli.scene_variant}{rig_tag}_{ckpt_tag}"
    run_evaluation(args_cli, policy=label, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    finally:
        stop_policy_server(server_proc)
