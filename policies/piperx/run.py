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
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.piperx.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_piperx_delta_envs,
    auto_register_piperx_envs,
    resolve_background,
)
from policies.piperx.client import PiperXOpenpiClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

_register = auto_register_piperx_envs if args_cli.action_space == "jointpos" else auto_register_piperx_delta_envs
_register(task_dirs=args_cli.task_dirs, task=args_cli.task, background_cfg=resolve_background(args_cli.background))


def make_client(args: argparse.Namespace) -> PiperXOpenpiClient:
    return PiperXOpenpiClient(
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
