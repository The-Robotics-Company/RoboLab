# food_packing: pi05 zero-shot vs fine-tune pipeline

Task: *"put the mustard bottle in the left bin and the spam can in the right bin"* on the
RoboLab `food_packing_opus_gt` scene, Franka + Robotiq 2F-85 (Nucleus `franka.usd`, Gripper variant).

| file | role |
|---|---|
| `scene_physics.py` | every physics override (`apply_scene_physics`), the shared CLI flags (`add_physics_args`), the success test (`in_bin`), wrist-camera presets (`WRIST_CAM`) |
| `scene_cfg.py` | the N-env `InteractiveScene` (robot, two cameras, all 9 objects registered) |
| `gen_demos.py` + `pick_sequence.py` | vectorised RMPflow expert; `--dev` hot-reloads the waypoints |
| `eval.py` | vectorised policy eval against an openpi websocket server (20 eps ≈ 4 min) |
| `make_dr_episodes.py` → `dr_episodes.json` | shared DR draws: eval reads 0–19, demos 20+ |
| `robotiq_loop.py` | Robotiq four-bar loop-closure joints (without them nothing can be grasped) |
| `probe_wrist_cam.py` | renders candidate wrist-camera poses in one Isaac boot |
| `grasps/` | object-frame grasps recorded with `trc-rollout examples/food_packing_pick.py --record-grasp` |

## Run

```bash
cd RoboLab && export DISPLAY=:1 PYTHONPATH=$PWD          # --headless hangs on the TRC box
F=policies/pi0_family/foodpacking
python $F/gen_demos.py --num-envs 16 --episodes 160 --dr-start 20 --wrist-cam side --out ~/datasets/fp_demos
# convert + train: openpi/examples/robolab/npz_to_lerobot.py, config pi05_robolab_franka_lora
# serve: cd openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/serve_policy.py policy:checkpoint --policy.config=<cfg> --policy.dir=<ckpt>
python $F/eval.py --episodes 20 --num-envs 20 --obs-format droid   --home-from-demos ~/datasets/fp_demos --wrist-cam side   # zero-shot (pi05_droid_jointpos)
python $F/eval.py --episodes 20 --num-envs 20 --obs-format robolab --home-from-demos ~/datasets/fp_demos --wrist-cam side   # fine-tuned (pi05_robolab_franka*)
```

Train and eval **must** share: physics flags, `--wrist-cam`, render size (320x180), start-pose
source (`--home-from-demos`), and the DR file.

## Physics decisions (all measured, 2026-09-14)

| setting | why |
|---|---|
| all 9 objects dynamic except the 2 bins (kinematic) | dynamic bins: a dropped can launches the bin at 10 m/s (7/16 vs 16/16 kept) |
| distractors → single `convexHull` | the delivered 64-hull decompositions let the 60 g box sink into the table and the can tip off it; 9/16 envs disturbed before the arm moved, ~1/16 exploded |
| bins → `convexDecomposition` (not SDF) | SDF exploded and hung multi-env startup |
| `maxDepenetrationVelocity` 0.5 on objects | delivered poses sit mm inside each other |
| finger pads friction 1.5, graspables 64/8 solver iters | required: 0/16 grasps without |
| loop-closure D6 joints | required: 0/16 without |
| start pose held during settle | otherwise the arm drifts to the last target |
| success = centre inside measured inner bin footprint, below rim | 13 cm radius rejected corner placements |

## Gotchas

* Isaac `app.close()` hangs here; scripts `os._exit(0)`. Two `<defunct>` helpers per Isaac
  process are reaped in-process (`reap_zombie_children`).
* Kill by pid. `pkill -f <pattern>` kills the shell that spells the pattern.
* The `legacy` wrist preset (RoboLab DroidCfg offset) looks into the gripper housing on this
  asset: 80% grey. Use `side`.
* DR sampler leaves the mustard within 5 mm of the container in 193/200 draws (rule
  "no closer than nominal"). Known, not yet changed.
* Residual ~2% robot articulation blow-up (5200 Nm sim effort limits). Counted as failure.
