# food_packing: pi05 zero-shot vs fine-tune pipeline

Task: *"put the mustard bottle in the left bin and the spam can in the right bin"* on the
RoboLab `food_packing_opus_gt` scene, Franka + Robotiq 2F-85 (Nucleus `franka.usd`, Gripper variant).

| file | role |
|---|---|
| `scene_physics.py` | every physics override (`apply_scene_physics`), the shared CLI flags (`add_physics_args`), the success test (`in_bin`), wrist-camera presets (`WRIST_CAM`), scene registry + per-scene bin geometry (`SCENES`, `BIN_GEOM`, `select_scene`) |
| `scene_cfg.py` | the N-env `InteractiveScene` (robot, two cameras, all 9 objects registered) |
| `gen_demos.py` + `pick_sequence.py` | vectorised RMPflow expert; `--dev` hot-reloads the waypoints |
| `eval.py` | vectorised policy eval against an openpi websocket server (20 eps ≈ 4 min) |
| `make_dr_episodes.py` → `dr_episodes.json` | shared DR draws: eval reads 0–19, demos 20+ |
| `robotiq_loop.py` | Robotiq four-bar loop-closure joints (without them nothing can be grasped) |
| `probe_wrist_cam.py` | renders candidate wrist-camera poses in one Isaac boot |
| `make_compare_grid.py`, `make_demo_video.py` | result grid video ({zero-shot, fine-tune} x {recon, GT}) and side-by-side training-demo video from the npz files; openpi venv, no Isaac |
| `grasps/` | object-frame grasps recorded with `trc-rollout examples/food_packing_pick.py --record-grasp` |


The RoboLab task class (`FoodPackingTask`, used by `examples/run_food_packing.py` and `--task FoodPackingTask` in
the pi0_family runners) lives in `robolab/tasks/trc/food_packing.py`, NOT in `robolab/tasks/benchmark/`: the
`trc` folder is outside `DEFAULT_TASK_SUBFOLDERS`, so nothing here is registered or validated by default. Opt in with
`--task food_packing.py` (a file name is searched under all of `robolab/tasks/`; a bare class name only searches
`benchmark/`), or `--task-dirs trc` for discovery-based registration.

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

## Scenes (`--scene`)

| name | file | what |
|---|---|---|
| `opus_gt` (default) | `assets/scenes/food_packing_opus_gt.usda` | Robolab delivery-adapter reconstruction; every demo and result before 2026-09-15 is on it |
| `gt` | `assets/scenes/food_packing_gt_canon.usda` | RoboLab's authored `food_packing.usda` (ycb + vomp assets, Xuning Yang) with prims renamed to the canonical names and the table pinned kinematic. Generated from `food_packing.usda`; poses byte-identical |

Same robot, cameras, DR file, start poses, physics flags and success test; only table + objects
differ. Train and eval MUST use the same `--scene`. GT assets are git-LFS pointers until
`git lfs pull --include="assets/objects/ycb/**,assets/objects/vomp/bin_a06/**,assets/objects/vomp/bin_b03/**"`
(~250 MB); without it the scene loads as a bare table.

GT bin geometry (`BIN_GEOM["gt"]`, from mesh vertices x scene scale): `bin_a06` outer 0.356 x 0.210,
rim +0.131 (front lip +0.104); `bin_b03` outer 0.324 x 0.196, rim +0.168 (front lip +0.069). Both
are open-front stacking bins, lip toward the robot, roots at the mesh centre.

Name map opus_gt -> gt source: grey_bin_left=bin_a06, grey_bin_right=bin_b03, cheezit_box=cheez_it,
soup_can=tomato_soup_can, mustard_bottle=mustard, container=coffee_can, small_box=chocolate_pudding.

### GT eval 2026-09-15 (reconstruction-trained policies on the GT scene)

Smoke test: `eval.py --dry-run --scene gt --num-envs 16 --max-steps 45` -> 16/16 stable, 0 blow-ups.
Then the two evals below, legacy wrist cam, demos v3 start poses, DR 0-19, 600-step cap:

```bash
python $F/eval.py --episodes 20 --num-envs 20 --scene gt --obs-format droid   --home-from-demos ~/Desktop/trc/datasets/food_packing_demos_v3 --wrist-cam legacy --video videos/gt_zeroshot.mp4 --out ~/Desktop/trc/eval_gt/zeroshot
python $F/eval.py --episodes 20 --num-envs 20 --scene gt --obs-format robolab --home-from-demos ~/Desktop/trc/datasets/food_packing_demos_v3 --wrist-cam legacy --video videos/gt_finetune.mp4 --out ~/Desktop/trc/eval_gt/finetune
```

| policy | scene | full | mustard | spam | blow-ups |
|---|---|---|---|---|---|
| zero-shot `pi05_droid_jointpos` | opus_gt | 0/20 | 0 | 0 | 0 |
| LoRA foodpacking_v2 step 2999 | opus_gt | 5/20 | 12 | 5 | 0 |
| zero-shot `pi05_droid_jointpos` | gt | 0/20 | 1 | 0 | 0 |
| LoRA foodpacking_v2 step 2999 | gt | 0/20 | 11 | 0 | 0 |

Comparison video: `videos/compare_grid_final.mp4` (4 columns, sub-scores; built by `make_compare_grid.py`, sources `zs_vec`, `ft_vec`, `gt_zeroshot`, `gt_finetune`; `--eps 4 --cell 480 --panel 0 --out compare_grid_final_ep4.mp4` for the single-row cut).
Mustard transfers (11 vs 12); spam does not (0 vs 5): the spam is nudged in 7/20 GT episodes
(displacement 0.08-0.21 m) but never lifted. Videos: `videos/gt_zeroshot*.mp4`, `videos/gt_finetune*.mp4`;
results: `~/Desktop/trc/eval_gt/{zeroshot,finetune}/results.json`.

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
