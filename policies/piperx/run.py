# SPDX-License-Identifier: Apache-2.0
"""Evaluate an openpi Piper X policy (served by scripts/serve_policy.py) on RoboLab tasks with the Piper X registration.

Example (server on :8001 serving pi05_piperx_rubiks_expert_5k):
    uv run python policies/piperx/run.py --headless --num-envs 10 --task RubiksCubeTask \
        --remote-port 8001 --prompt "put the rubiks cube in the bowl" --output-folder-name piperx_rubiks_step1000
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an openpi Piper X policy in RoboLab.")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000)
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None)
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=15,
                    help="Actions executed per server query (= training action_horizon for pi05_piperx*).")
parser.add_argument("--prompt", type=str, default=None,
                    help="Override the task's instruction with the exact training prompt.")
parser.add_argument("--gripper-close-above", type=float, default=0.22)
parser.add_argument("--gripper-open-below", type=float, default=0.10)
parser.add_argument("--action-space", choices=["jointpos", "delta"], default="jointpos",
                    help="Piper X registration: absolute joint targets (default) or per-step deltas.")
parser.add_argument("--background", type=str, default="home_office",
                    help="Piper X registration background preset (see registrations/piperx).")
parser.add_argument("--client", choices=["piperx", "droid"], default="piperx",
                    help=("piperx: a policy trained with LeRobotRoboLabDataConfig (6 joints + gripper). "
                          "droid: the un-fine-tuned pi05_droid_jointpos checkpoint, dimensions adapted "
                          "from the Franka's 7 joints -- a control, not a working policy."))
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")
parser.add_argument("--cube-pose-set", "--cube_pose_set", type=str, default=None,
                    help=("JSON file of fixed per-env cube XY offsets (metres). env i always gets offset i, "
                          "so every checkpoint and scene variant is scored on an identical set of object "
                          "poses. Without it the cube stays at the scene's authored pose."))
parser.add_argument("--scene-variant", "--scene_variant", type=str, default="gt",
                    help=("gt: the task's own RoboLab scene. Any other value swaps in the scene file with the "
                          "same name plus that suffix (rubiks_cube_bowl.usda -> rubiks_cube_bowl_v0.usda). "
                          "Everything else -- client, prompt, thresholds -- is unchanged, so a v0 run is "
                          "comparable to a gt run of the same checkpoint."))

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Retarget scene lookup before the registrations import below, because task classes resolve their scene at
# class-definition time and RoboLab executes task files fresh at registration.
# Same mechanism as policies/pi0_family/run_rollout.py::install_scene_variant, inlined because importing that
# module would run its own argparse and AppLauncher at import time.
if args_cli.scene_variant != "gt":
    import os as _os

    import robolab.core.scenes.utils as _scene_utils  # noqa: E402

    _orig_find = _scene_utils.find_scene_file
    _suffix = f"_{args_cli.scene_variant}"

    def _find_scene_file_variant(scene_path: str, scene_dir: str, *a, **kw) -> str:
        if _os.path.isabs(scene_path):
            return scene_path
        stem, ext = _os.path.splitext(scene_path)
        if stem.endswith(_suffix):
            return _orig_find(scene_path, scene_dir, *a, **kw)
        name = f"{stem}{_suffix}{ext}"
        resolved = _orig_find(name, scene_dir, *a, **kw)
        if not _os.path.isfile(resolved):
            raise FileNotFoundError(
                f"--scene-variant {args_cli.scene_variant}: no scene file '{name}' under {scene_dir} "
                f"for task scene '{scene_path}'.")
        print(f"[RoboLab] scene variant {args_cli.scene_variant}: {scene_path} -> {resolved}", flush=True)
        return resolved

    _scene_utils.find_scene_file = _find_scene_file_variant

import robolab.constants  # noqa: E402
from robolab.registrations.piperx.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_piperx_delta_envs,
    auto_register_piperx_envs,
    resolve_background,
)
from policies.piperx.client import PiperXDroidBaselineClient, PiperXOpenpiClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

# Inject a deterministic object-pose event before registration. The task classes are imported here first,
# so the attribute we set is the same class object the registration will pick up.
if args_cli.cube_pose_set:
    import json as _json

    from isaaclab.managers import EventTermCfg as _EventTerm
    from isaaclab.utils import configclass as _configclass

    from robolab.core.events.reset_pose_fixed import reset_pose_fixed_table as _fixed
    from robolab.core.task.task import Task as _Task
    from robolab.tasks.benchmark.rubiks_cube_task import RubiksCubeTask as _RCT

    _spec = _json.loads(open(args_cli.cube_pose_set).read())
    _offsets = _spec["offsets"]

    @_configclass
    class _FixedCubePose:
        randomize_init_pose = _EventTerm(
            func=_fixed, mode="reset",
            params={"asset_cfg": ["rubiks_cube"], "offsets": _offsets, "reset_to_default_otherwise": True},
        )

    # Set it on the BASE class as well: registration discovers and re-executes task files, so the concrete
    # class object we patch here is thrown away and rebuilt. Inheritance survives that; the first smoke test
    # silently ran with the cube at its authored pose because only the concrete class was patched.
    _Task.events = _FixedCubePose
    _RCT.events = _FixedCubePose
    print(f"[RoboLab] fixed cube pose set: {len(_offsets)} offsets from {args_cli.cube_pose_set}", flush=True)

_register = auto_register_piperx_envs if args_cli.action_space == "jointpos" else auto_register_piperx_delta_envs
_register(task_dirs=args_cli.task_dirs, task=args_cli.task, background_cfg=resolve_background(args_cli.background))


def make_client(args: argparse.Namespace) -> PiperXOpenpiClient:
    cls = PiperXDroidBaselineClient if args.client == "droid" else PiperXOpenpiClient
    return cls(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        prompt_override=args.prompt,
        gripper_close_above=args.gripper_close_above,
        gripper_open_below=args.gripper_open_below,
    )


def main() -> None:
    run_evaluation(args_cli, policy="pi05_piperx", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
