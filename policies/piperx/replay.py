# SPDX-License-Identifier: Apache-2.0
"""Roll out the *recorded training actions* in the simulator, with no policy in the loop.

Answers "are the demonstrations any good" by executing them and letting RoboLab's own success
predicate judge the result, exactly as it judges a policy.

Two gripper modes, because the data and the env disagree about what the gripper channel means:
  --gripper raw          feed the recorded command straight through. The env's action term closes
                         only above 0.5 (BinaryJointPositionZeroToOneAction), and every episode
                         commands 0.966 for ~15 frames and then 0.27-0.33 for the whole transport,
                         so this releases the cube mid-carry. Tests replayability as-is.
  --gripper hysteresis   apply the same close>0.22 / open<0.10 hysteresis the eval client applies to
                         policy output. Tests whether the recorded *trajectory* achieves the task.

Example:
    uv run python policies/piperx/replay.py --headless --num-runs 1 --task RubiksCubeTask \
        --episodes 0,18,36,54,72,90 --gripper hysteresis --output-folder-name demo_replay_hyst
"""

import argparse
import pathlib
import sys
import traceback

import cv2  # noqa: F401 -- must import before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay recorded Piper X demonstrations in RoboLab.")
parser.add_argument("--dataset", type=str, required=True,
                    help="LeRobot v2.1 dataset root (episode parquets under data/chunk-000).")
parser.add_argument("--episodes", type=str, default="0",
                    help="Comma-separated episode indices; env_id i replays episodes[i].")
parser.add_argument("--gripper", choices=["raw", "hysteresis"], default="hysteresis")
parser.add_argument("--gripper-close-above", type=float, default=0.22)
parser.add_argument("--gripper-open-below", type=float, default=0.10)
parser.add_argument("--action-space", choices=["jointpos", "delta"], default="jointpos")
parser.add_argument("--background", type=str, default="home_office")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
EPISODES = [int(x) for x in args_cli.episodes.split(",")]
args_cli.num_envs = len(EPISODES)          # env_id i <-> EPISODES[i]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.eval.base_client import InferenceClient  # noqa: E402
from robolab.registrations.piperx.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_piperx_delta_envs,
    auto_register_piperx_envs,
    resolve_background,
)

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

_register = auto_register_piperx_envs if args_cli.action_space == "jointpos" else auto_register_piperx_delta_envs
_register(task_dirs=args_cli.task_dirs, task=args_cli.task, background_cfg=resolve_background(args_cli.background))


class DemoReplayClient(InferenceClient):
    """Returns recorded actions instead of querying a policy server."""

    def __init__(self, dataset: str, episodes: list[int], mode: str,
                 close_above: float, open_below: float,
                 exo_key: str = "exo_camera", wrist_key: str = "wrist_camera") -> None:
        super().__init__()
        root = pathlib.Path(dataset)
        self.exo_key, self.wrist_key = exo_key, wrist_key
        self.mode, self.close_above, self.open_below = mode, close_above, open_below
        self.actions: dict[int, np.ndarray] = {}
        for env_id, ep in enumerate(episodes):
            f = root / f"data/chunk-000/episode_{ep:06d}.parquet"
            a = np.stack(pq.read_table(f, columns=["actions"]).to_pandas()["actions"].values)
            self.actions[env_id] = a.astype(np.float32)
            print(f"[replay] env {env_id} <- episode {ep}: {len(a)} recorded actions")
        self._step: dict[int, int] = {}
        self._closed: dict[int, bool] = {}

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._step.clear(); self._closed.clear()
        else:
            self._step.pop(env_id, None); self._closed.pop(env_id, None)
        super().reset(env_id=env_id)

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        a = self.actions[env_id]
        i = self._step.get(env_id, 0)
        # Episodes are ~139 frames but the eval runs to its own horizon: hold the final recorded
        # action so the arm parks in the release pose instead of jumping to zero.
        action = np.array(a[min(i, len(a) - 1)], dtype=np.float32)
        self._step[env_id] = i + 1
        if self.mode == "hysteresis":
            closed = self._closed.get(env_id, False)
            g = float(action[6])
            if g > self.close_above:
                closed = True
            elif g < self.open_below:
                closed = False
            self._closed[env_id] = closed
            action[6] = 1.0 if closed else 0.0
        return {"action": action, "viz": self._build_visualization(
            self._extract_observation(obs, env_id=env_id))}

    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        images = self._find_obs_term(raw_obs, self.exo_key), self._find_obs_term(raw_obs, self.wrist_key)
        return {"exo": self._to_numpy(images[0], env_id) if images[0] is not None else None,
                "wrist": self._to_numpy(images[1], env_id) if images[1] is not None else None}

    def _build_visualization(self, extracted_obs: dict):
        a, b = extracted_obs.get("exo"), extracted_obs.get("wrist")
        if a is None or b is None:
            return None
        from openpi_client import image_tools
        return np.concatenate([image_tools.resize_with_pad(a, 224, 224),
                               image_tools.resize_with_pad(b, 224, 224)], axis=1)

    # Unused: there is no server. Present because the base class declares them abstract.
    def _pack_request(self, extracted_obs: dict, instruction: str):
        raise NotImplementedError

    def _query_server(self, request):
        raise NotImplementedError

    def _unpack_response(self, response) -> np.ndarray:
        raise NotImplementedError


def make_client(args: argparse.Namespace) -> DemoReplayClient:
    return DemoReplayClient(args.dataset, EPISODES, args.gripper,
                            args.gripper_close_above, args.gripper_open_below)


def main() -> None:
    run_evaluation(args_cli, policy="demo_replay", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
