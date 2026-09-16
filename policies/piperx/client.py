# SPDX-License-Identifier: Apache-2.0
"""Inference client for openpi policies trained with `LeRobotRoboLabDataConfig` (Piper X, 6 joints + gripper).

Talks to an openpi websocket policy server (scripts/serve_policy.py) serving e.g. `pi05_piperx_rubiks_expert_5k`.
Request keys match `openpi.policies.robolab_policy.RoboLabInputs`; the response is a (horizon, 7) chunk of absolute
joint targets (the server already converted deltas back) plus a continuous gripper in RoboLab polarity (0 open .. 1 closed).

Gripper: the cuRobo-generated training data commands ~0.18 while approaching, ~0.3 while holding and ~0.97 while
closing, so DROID's 0.5 threshold would never close. Default: close above 0.22, reopen below 0.10 (hysteresis per env).
"""

import atexit
import json
import logging
import os

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)

# Opt-in raw-gripper recorder (off unless PIPERX_GRIPPER_LOG is set), appended one JSON line per chunk.
_GRIPPER_LOG = open(os.environ["PIPERX_GRIPPER_LOG"], "a") if os.environ.get("PIPERX_GRIPPER_LOG") else None
if _GRIPPER_LOG is not None:
    atexit.register(_GRIPPER_LOG.close)


class PiperXOpenpiClient(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        remote_uri: str | None = None,
        open_loop_horizon: int = 15,
        prompt_override: str | None = None,
        gripper_close_above: float = 0.22,
        gripper_open_below: float = 0.10,
        exo_key: str = "exo_camera",
        wrist_key: str = "wrist_camera",
    ) -> None:
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon)
        self.prompt_override = prompt_override
        self.gripper_close_above = gripper_close_above
        self.gripper_open_below = gripper_open_below
        self.exo_key, self.wrist_key = exo_key, wrist_key
        self._gripper_closed: dict[int, bool] = {}
        self._current_env: int = 0
        self._remote = remote_uri if remote_uri is not None else (remote_host, remote_port)
        print(f"[{self.__class__.__name__}] Awaiting for server on {self._remote} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected.")

    def _connect(self):
        if isinstance(self._remote, str):
            return websocket_client_policy.WebsocketClientPolicy(self._remote)
        return websocket_client_policy.WebsocketClientPolicy(*self._remote)

    # ---- episode bookkeeping ------------------------------------------

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._gripper_closed.clear()
        else:
            self._gripper_closed.pop(env_id, None)

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        self._current_env = env_id  # lets _postprocess_chunk keep per-env hysteresis state
        return super().infer(obs, instruction, env_id=env_id)

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        images = raw_obs["image_obs"]
        proprio = raw_obs["proprio_obs"]
        return {
            "exo": self._to_numpy(images[self.exo_key], env_id),
            "wrist": self._to_numpy(images[self.wrist_key], env_id),
            "joint_position": self._to_numpy(proprio["arm_joint_pos"], env_id).astype(np.float32),
            "gripper_position": np.atleast_1d(self._to_numpy(proprio["gripper_pos"], env_id)).astype(np.float32),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "observation/image": image_tools.resize_with_pad(extracted_obs["exo"], 224, 224),
            "observation/wrist_image": image_tools.resize_with_pad(extracted_obs["wrist"], 224, 224),
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": self.prompt_override or instruction,
        }

    def _query_server(self, request: dict) -> dict:
        import websockets.exceptions

        for attempt in range(3):
            try:
                return self.client.infer(request)
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK, OSError) as e:
                if attempt == 2:
                    raise
                logger.warning("[%s] connection lost (%s), reconnecting", self.__class__.__name__, e)
                self.client = self._connect()
                self._chunks.clear()
                self._counters.clear()

    def _unpack_response(self, response: dict) -> np.ndarray:
        return np.asarray(response["actions"], dtype=np.float32)

    # ---- optional hooks -----------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Binarise the continuous gripper with hysteresis, walking through the chunk in time order."""
        chunk = chunk.copy()
        closed = self._gripper_closed.get(self._current_env, False)
        raw = []
        for t in range(chunk.shape[0]):
            g = float(chunk[t, -1])
            raw.append(g)
            if closed and g < self.gripper_open_below:
                closed = False
            elif not closed and g > self.gripper_close_above:
                closed = True
            chunk[t, -1] = 1.0 if closed else 0.0
        self._gripper_closed[self._current_env] = closed
        # The raw gripper prediction is only visible here, before binarisation. Set PIPERX_GRIPPER_LOG to
        # record it: the margin between it and gripper_close_above is what decides a premature grasp.
        if _GRIPPER_LOG is not None:
            _GRIPPER_LOG.write(json.dumps({"env": int(self._current_env), "closed_after": bool(closed),
                                           "raw": [round(v, 5) for v in raw]}) + "\n")
            _GRIPPER_LOG.flush()
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        a = image_tools.resize_with_pad(extracted_obs["exo"], 224, 224)
        b = image_tools.resize_with_pad(extracted_obs["wrist"], 224, 224)
        return np.concatenate([a, b], axis=1)


if __name__ == "__main__":
    # Protocol smoke test against a running server: fake observation, 16 inferences (1 server call per horizon).
    import argparse
    import time

    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-port", type=int, default=8000)
    a = ap.parse_args()
    client = PiperXOpenpiClient(remote_port=a.remote_port, prompt_override="put the rubiks cube in the bowl")
    fake_obs = {
        "image_obs": {
            "exo_camera": torch.zeros((1, 352, 624, 3), dtype=torch.uint8),
            "wrist_camera": torch.zeros((1, 352, 624, 3), dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.tensor([[0.0, 1.2, -1.2, 0.0, 0.0, 0.0]]),
            "gripper_pos": torch.zeros((1, 1)),
        },
    }
    t0 = time.time()
    for _ in range(16):
        out = client.infer(fake_obs, "put the rubiks cube in the bowl", env_id=0)
    print("action", np.round(out["action"], 3), f"| 16 steps in {time.time() - t0:.1f}s")


class PiperXDroidBaselineClient(PiperXOpenpiClient):
    """Drive the Piper X with the UN-fine-tuned pi05_droid_jointpos checkpoint (the RoboLab DROID policy).

    A control, to show what the off-the-shelf DROID policy does on a robot it was never trained on. The
    checkpoint speaks the Franka's 7 arm joints, the Piper has 6, so the dimensions are adapted:
      observation/joint_position  6 -> 7   (a zero appended for the Franka's extra joint)
      actions                     8 -> 7   (first 6 joint targets kept, plus the gripper)
    Its DROID norm stats are Franka joint ranges, so the Piper's joint values normalise to something the
    checkpoint never saw. That is the point of the control, not a bug to fix.

    Gripper: DROID's convention is already 0 = open .. 1 = closed (robolab/robots/droid.py gripper_pos), the
    same as the Piper config, so no polarity flip -- only DROID's standard 0.5 binarisation.
    """

    N_FRANKA_JOINTS = 7
    N_PIPER_JOINTS = 6

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        # DROID's own key names (openpi.policies.droid_policy.DroidInputs), not the RoboLab ones.
        jp = np.asarray(extracted_obs["joint_position"], dtype=np.float32).reshape(-1)
        pad = self.N_FRANKA_JOINTS - jp.shape[0]
        if pad > 0:
            jp = np.concatenate([jp, np.zeros(pad, dtype=np.float32)])
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(extracted_obs["exo"], 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(extracted_obs["wrist"], 224, 224),
            "observation/joint_position": jp[: self.N_FRANKA_JOINTS],
            "observation/gripper_position": np.atleast_1d(
                np.asarray(extracted_obs["gripper_position"], dtype=np.float32)
            ),
            "prompt": self.prompt_override or instruction,
        }

    def _unpack_response(self, response: dict) -> np.ndarray:
        a = np.asarray(response["actions"], dtype=np.float32)
        # (horizon, 8) Franka -> (horizon, 7) Piper: drop the 7th joint column, keep the gripper.
        return np.concatenate([a[:, : self.N_PIPER_JOINTS], a[:, -1:]], axis=-1)

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        # DROID's own execution rule: hard threshold at 0.5, no hysteresis.
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk
