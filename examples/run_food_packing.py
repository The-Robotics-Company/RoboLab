# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Open the food_packing scene with the Franka arm, gripper and its table/stand.

The scene USD carries only the groceries and the wooden work table. The arm and
the fixture it is bolted to are added here, by the env factory: ``FrankaCfg``
spawns ``panda_instanceable.usd`` at the env origin and declares
``FRANKA_TABLE_FIXTURE``, which the factory spawns as ``table_fixture`` (the
scene's own ``franka_table`` prim is deactivated and replaced).

Usage:
    python examples/run_food_packing.py                 # open and hold for inspection
    python examples/run_food_packing.py --num-steps 240 # settle physics, then hold
    python examples/run_food_packing.py --headless
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401  must be imported before isaaclab

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="FoodPackingTask")
parser.add_argument("--num-steps", type=int, default=0, help="physics steps to run before holding")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import (  # noqa: E402
    generate_image_obs_from_cameras,
    generate_obs_cfg,
)
from robolab.constants import TASK_DIR  # noqa: E402
from robolab.variations.camera import EgocentricWideAngleCameraCfg  # noqa: E402
from robolab.variations.lighting import SphereLightCfg  # noqa: E402

robolab.constants.RECORD_IMAGE_DATA = False


def register_franka_env(task: str) -> str:
    """Register ``task`` against the Franka, mirroring auto_register_example_envs_franka."""
    from robolab.robots.franka import FrankaCfg, FrankaJointPositionActionCfg, contact_gripper

    camera_cfgs = [EgocentricWideAngleCameraCfg]
    ImageObsCfg = generate_image_obs_from_cameras(camera_cfgs)
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg()})

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        tasks=task,
        add_tags=["benchmark"],
        env_prefix="",
        env_postfix="FrankaJointPosition",
        observations_cfg=ObservationCfg(),
        actions_cfg=FrankaJointPositionActionCfg(),
        robot_cfg=FrankaCfg,
        camera_cfg=camera_cfgs,
        lighting_cfg=SphereLightCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )
    envs = get_envs(task=task)
    if not envs:
        raise RuntimeError(f"no env registered for task {task!r}")
    return envs[0]


def main() -> None:
    env_name = register_franka_env(args_cli.task)
    print(f"[food_packing] env: {env_name}", flush=True)

    env, _ = create_env(env_name, device=args_cli.device, num_envs=1, use_fabric=True)
    env.reset()

    robot = env.scene["robot"]
    print(f"[food_packing] robot bodies: {len(robot.data.body_names)}", flush=True)
    print(f"[food_packing] robot joints: {robot.data.joint_names}", flush=True)
    print(f"[food_packing] action dim:   {env.action_manager.total_action_dim}", flush=True)
    print("[food_packing] scene ready -- arm, gripper and table/stand are in the stage", flush=True)

    if args_cli.num_steps:
        import torch

        hold = torch.cat([
            robot.data.default_joint_pos[0, :7],
            torch.zeros(1, device=env.device),
        ]).unsqueeze(0)
        for _ in range(args_cli.num_steps):
            env.step(hold)
        print(f"[food_packing] settled {args_cli.num_steps} steps", flush=True)

    while simulation_app.is_running():
        env.sim.render()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
