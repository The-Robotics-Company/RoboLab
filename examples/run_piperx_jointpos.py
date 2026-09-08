# SPDX-License-Identifier: Apache-2.0

# isort: skip_file

"""Rendered joint-position smoke test for the fixed-base PiPER-X on RubiksCubeTask.

Checks: env builds from PiperXCfg, action dim is 7 (6 arm + gripper), the gripper tracks the binary command
with both fingers mirrored, the arm tracks a small sinusoid around the home pose, and the exo + wrist camera
observations arrive at 624x352. Writes the viewport video and a three-panel diagnostic video.

  cd ~/git/RoboLab && .venv/bin/python examples/run_piperx_jointpos.py --headless
  PIPERX_WRIST_CAM=calibrated .venv/bin/python examples/run_piperx_jointpos.py --headless   # hand-eye camera pose
"""

import argparse
import math
import os
import sys
import traceback

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=180)
parser.add_argument("--task", type=str, default="RubiksCubeTask")
parser.add_argument("--delta", action="store_true", help="use the delta joint-position registration")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

from robolab.constants import PACKAGE_DIR  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.robots.piper_x import (  # noqa: E402
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_COMMANDS_CLOSE,
    GRIPPER_JOINT_COMMANDS_OPEN,
    WRIST_CAM_SOURCE,
)
from robolab.registrations.piperx.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_piperx_delta_envs,
    auto_register_piperx_envs,
)


def compose_diagnostic_frame(policy_images, viewport):
    views = [("EXO (DC1)", policy_images["exo_camera"]), ("WRIST (DC1)", policy_images["wrist_camera"]), ("VIEWPORT", viewport)]
    h = max(v.shape[0] for _, v in views)
    labeled = []
    for label, view in views:
        img = view.copy()
        if img.shape[0] != h:  # scale to common height for hconcat
            img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        labeled.append(img)
    return cv2.hconcat(labeled)


def main() -> None:
    if args_cli.delta:
        auto_register_piperx_delta_envs(task=args_cli.task)
        postfix = "PiperXDeltaJointPosition"
    else:
        auto_register_piperx_envs(task=args_cli.task)
        postfix = "PiperXJointPosition"
    env_name = next(n for n in get_envs(task=args_cli.task) if n.endswith(postfix))
    env, env_cfg = create_env(env_name, device=args_cli.device, num_envs=1, use_fabric=True)
    output_dir = os.path.join(PACKAGE_DIR, "output", "piperx_jointpos_smoke")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"piperx_{postfix}.mp4")
    diagnostic_path = os.path.join(output_dir, f"piperx_{postfix}_three_camera_diagnostic.mp4")
    fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    video = VideoWriter(video_path, fps)
    diagnostic_video = VideoWriter(diagnostic_path, fps)

    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        arm_idx = [robot.data.joint_names.index(n) for n in ARM_JOINT_NAMES]
        home = robot.data.default_joint_pos[0, arm_idx].clone()
        print(f"Environment: {env_name}  (wrist camera source: {WRIST_CAM_SOURCE})")
        print(f"Bodies: {robot.data.body_names}")
        print(f"Joints: {robot.data.joint_names}")
        print(f"Action dimension: {env.action_manager.total_action_dim}")
        images = unpack_image_obs(obs)
        for k, v in images.items():
            print(f"image obs {k}: {tuple(v.shape)}")
        assert env.action_manager.total_action_dim == 7, env.action_manager.total_action_dim

        gripper_error = None
        for step in range(args_cli.num_steps):
            phase = 2.0 * math.pi * step / args_cli.num_steps
            if args_cli.delta:
                arm_cmd = torch.zeros(6, device=env.device)
                arm_cmd[0] = 0.01 * math.cos(phase)     # small delta per step
                arm_cmd[5] = 0.01 * math.cos(2.0 * phase)
            else:
                arm_cmd = home.clone()
                arm_cmd[0] += 0.25 * math.sin(phase)
                arm_cmd[5] += 0.15 * math.sin(2.0 * phase)
            gripper_command = 1.0 if math.sin(phase) > 0.0 else 0.0
            action = torch.cat([arm_cmd, torch.tensor([gripper_command], device=env.device)]).unsqueeze(0)
            obs, _, _, _, _ = env.step(action)
            if step == args_cli.num_steps // 2 - 1:  # gripper has been "close" for a quarter period
                jp = dict(zip(robot.data.joint_names, robot.data.joint_pos[0].tolist()))
                gripper_error = max(abs(jp[n] - t) for n, t in GRIPPER_JOINT_COMMANDS_CLOSE.items())
                mirror_error = abs(jp["gripper_joint1"] + jp["gripper_joint2"])
                print(f"gripper at close: {jp['gripper_joint1']:.4f} / {jp['gripper_joint2']:.4f} m, mirror error {mirror_error:.5f} m")
            policy_images = unpack_image_obs(obs)
            frame = unpack_viewport_cams(obs).get("combined_image")
            if frame is not None:
                video.write(frame)
                diagnostic_video.write(compose_diagnostic_frame(policy_images, frame))

        jp = dict(zip(robot.data.joint_names, robot.data.joint_pos[0].tolist()))
        open_error = max(abs(jp[n] - t) for n, t in GRIPPER_JOINT_COMMANDS_OPEN.items())
        print(f"gripper close error {gripper_error:.5f} m, open error {open_error:.5f} m")
        if not args_cli.delta:
            tracking_error = torch.max(torch.abs(robot.data.joint_pos[0, arm_idx] - arm_cmd)).item()
            print(f"Final maximum arm tracking error: {tracking_error:.5f} rad")
        print(f"Saved video: {video_path}")
        print(f"Saved diagnostic video: {diagnostic_path}")
    finally:
        video.release()
        diagnostic_video.release()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Terminated with error: {error}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
