# SPDX-License-Identifier: Apache-2.0
"""Deterministic object placement from a fixed table of offsets.

Why not just seed ``reset_pose_uniform``: a seeded sampler only reproduces a sequence if the number and
order of random draws is identical, which is not something an evaluation can guarantee across checkpoints,
scene variants or IsaacLab versions. Indexing a table by ``env_id`` is reproducible by construction -- env 3
gets offset 3 in every run, forever -- which is what makes success rates comparable across checkpoints.

Offsets are applied to each scene's OWN default pose, so the same table expresses the same relative
perturbation in the authored scene and in a reconstruction whose object rests somewhere slightly different.
"""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv

from robolab.core.events.reset_pose import _get_all_asset_names, _parse_asset_cfg, _reset_assets_to_default


def reset_pose_fixed_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg,
    offsets: list,
    reset_to_default_otherwise: bool = True,
):
    """Place each named asset at its default pose plus ``offsets[env_id % len(offsets)]``.

    Args:
        offsets: list of (dx, dy) or (dx, dy, dz) in metres, indexed by environment id.
    """
    asset_names = _parse_asset_cfg(asset_cfg)
    if reset_to_default_otherwise:
        others = _get_all_asset_names(env) - set(asset_names)
        if others:
            _reset_assets_to_default(env, env_ids, others)

    table = torch.tensor([[float(o[0]), float(o[1]), float(o[2]) if len(o) > 2 else 0.0] for o in offsets],
                         dtype=torch.float32, device=env.device)

    for name in asset_names:
        asset: RigidObject | Articulation = env.scene[name]
        state = asset.data.default_root_state[env_ids].clone()
        state[:, 0:3] += env.scene.env_origins[env_ids]
        idx = torch.as_tensor([int(e) % table.shape[0] for e in env_ids], device=env.device)
        state[:, 0:3] += table[idx]
        asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]), env_ids=env_ids)
        print(f"[RoboLab] fixed pose set: {name} offsets applied for envs {[int(e) for e in env_ids]}", flush=True)
