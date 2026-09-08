# SPDX-License-Identifier: Apache-2.0

"""Pi0-family inference client for the PiPER-X (6 joints + gripper, exo + wrist Orbbec DC1 cameras).

Same server protocol as :class:`Pi0DroidJointposClient` (openpi websocket server); only the observation
extraction and request packing differ. The request keys must match the observation names the checkpoint
was trained with -- edit ``REQUEST_KEYS`` (or subclass) when your openpi config uses different names.
"""

import numpy as np
import torch
from openpi_client import image_tools

from policies.pi0_family.client import Pi0DroidJointposClient

# RoboLab observation names (from robolab.robots.piper_x / registrations.piperx)
EXO_OBS = "exo_camera"
WRIST_OBS = "wrist_camera"

# What the openpi server expects. Defaults mirror the DROID-style keys so a pi05_droid_jointpos-shaped config
# works out of the box; a Piper-X-trained config will usually rename these (e.g. observation/exo_image).
REQUEST_KEYS = {
    "exo_image": "observation/exterior_image_1_left",
    "wrist_image": "observation/wrist_image_left",
    "joint_position": "observation/joint_position",
    "gripper_position": "observation/gripper_position",
    "prompt": "prompt",
}
IMAGE_SIZE = 224  # pi0.5 resizes every input to 224x224 (resize_with_pad), so send 224 to save bandwidth


class Pi0PiperXClient(Pi0DroidJointposClient):
    """openpi websocket client speaking the Piper-X observation/action contract (7-dim actions).

    ``droid_compat=True`` lets a DROID-trained checkpoint (7-joint Franka, 8-dim actions) drive the 6-joint
    Piper-X for pipeline tests: the joint state is padded to 7 with a zero 7th joint (the DROID output
    transform adds the state to the predicted deltas, so it must be 7 wide), and the 7th joint action is
    dropped from the returned chunk. This is a plumbing shim, not a meaningful policy transfer.
    """

    ACTION_DIM = 7  # joint1..6 + gripper (0 open, 1 close)
    DROID_JOINTS = 7

    def __init__(self, *args, droid_compat: bool = False, **kwargs) -> None:
        self.droid_compat = droid_compat
        super().__init__(*args, **kwargs)
        if droid_compat:
            print(f"[{self.__class__.__name__}] DROID compat: padding joints 6->7 on input, dropping joint 7 on output.")

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        exo = raw_obs["image_obs"][EXO_OBS][env_id].clone().detach().cpu().numpy()
        wrist = raw_obs["image_obs"][WRIST_OBS][env_id].clone().detach().cpu().numpy()
        proprio = raw_obs["proprio_obs"]
        return {
            "exo_image": exo,
            "wrist_image": wrist,
            "joint_position": proprio["arm_joint_pos"][env_id].clone().detach().cpu().numpy(),      # (6,) rad
            "gripper_position": proprio["gripper_pos"][env_id].clone().detach().cpu().numpy(),      # (1,) 0 open..1 closed
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        k = REQUEST_KEYS
        joints = np.asarray(extracted_obs["joint_position"], dtype=np.float32)
        if self.droid_compat and joints.shape[-1] < self.DROID_JOINTS:
            joints = np.concatenate([joints, np.zeros(self.DROID_JOINTS - joints.shape[-1], dtype=np.float32)])
        return {
            k["exo_image"]: image_tools.resize_with_pad(extracted_obs["exo_image"], IMAGE_SIZE, IMAGE_SIZE),
            k["wrist_image"]: image_tools.resize_with_pad(extracted_obs["wrist_image"], IMAGE_SIZE, IMAGE_SIZE),
            k["joint_position"]: joints,
            k["gripper_position"]: extracted_obs["gripper_position"],
            k["prompt"]: instruction,
        }

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32).copy()
        if self.droid_compat and chunk.shape[-1] == self.DROID_JOINTS + 1:
            chunk = np.concatenate([chunk[..., :6], chunk[..., -1:]], axis=-1)  # drop Franka joint 7
        if chunk.shape[-1] != self.ACTION_DIM:
            raise ValueError(
                f"Server returned action dim {chunk.shape[-1]}, expected {self.ACTION_DIM} "
                "(joint1..6 + gripper). Check the checkpoint's action space / --policy-config."
            )
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)  # binary gripper
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        img1 = image_tools.resize_with_pad(extracted_obs["exo_image"], IMAGE_SIZE, IMAGE_SIZE)
        img2 = image_tools.resize_with_pad(extracted_obs["wrist_image"], IMAGE_SIZE, IMAGE_SIZE)
        return np.concatenate([img1, img2], axis=1)


if __name__ == "__main__":
    # Round-trip against a running server: python policies/pi0_family/piperx_client.py (used by run_rollout.py --robot piperx)
    client = Pi0PiperXClient()
    fake_obs = {
        "image_obs": {
            EXO_OBS: [torch.zeros((352, 624, 3), dtype=torch.uint8)],
            WRIST_OBS: [torch.zeros((352, 624, 3), dtype=torch.uint8)],
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 6), dtype=torch.float32),
            "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
        },
    }
    ret = client.infer(fake_obs, "put the cube in the bowl")
    print("action:", ret["action"].shape, ret["action"])
