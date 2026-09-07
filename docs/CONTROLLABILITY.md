# Controllability of the counterfactual guidance

`docs/REPRODUCIBILITY.md` covers the published protocol and artifact. This file covers a different
question: **does the counterfactual guidance actually control the scene, and by how much?** That is the
prerequisite for any downstream use — probing a planner, stress-testing a recorded ego, mining
safety-critical variants — and it is not answered by the paper's ADE/ORR/CR table, which a completely
inert guidance would also reproduce.

## Why this needed checking at all

As released, the guidance was a no-op under `--part_control`. Two independent defects had to both be
present, and both are fixed as of the commit that added this file:

1. **The optimiser was handed a detached copy.** Sampling runs under `torch.no_grad()`, so
   `x_guidance[..., available_idx, :ts]` evaluates to a *copy* that is a leaf with `requires_grad=False`.
   `torch.optim.Adam` accepts it silently, `backward()` populates `x_guidance.grad` instead, and
   `opt.step()` updates a tensor that is then discarded.
2. **The guidance targeted the wrong agents.** tbsim resolves `guide_cfg.agents` as
   `cur_scene_inds[agents]` — indices into the scene's own agent rows — but `scene_editor.py` passed
   positions within `control_idx`. For the published artifact the two index spaces disagree in 79 of 90
   scenes.

Either defect alone zeroes the update. Neither produces an error, a warning, or an implausible metric:
the guidance loss is still computed and printed, and it still varies from scene to scene.

**The only reliable check is where the gradient lands.** Set `COUNTERSCENE_DEBUG_GRAD=1` and every
optimisation step prints the agent rows carrying non-zero gradient next to `available_idx`. They must
intersect:

```
before  nonzero_rows=[1, 6]  available_idx=[0, 8]   max|delta| = 0.000000
after   nonzero_rows=[6, 8]  available_idx=[0, 8]   max|delta| = 0.300000
```

Run that probe on one scene before trusting any batch of results.

## Protocol

Five arms over one scene set, all with `--part_control --controllable_agent 2 --num_scenes_per_batch 1`,
50 simulation steps, `n_step_action 5`:

| arm | guidance | seed | purpose |
|---|---|---|---|
| `baseline` | off | 0 | reference rollout everything else is scored against |
| `baseline_s0b` | off | 0 | same-seed replicate — pure run-to-run nondeterminism |
| `baseline_s1` | off | 1 | different-seed replicate — chaotic divergence floor |
| `conservative` | `very_conservative` | 0 | dose 1.2 |
| `full` | `full` | 0 | dose 3.0 (the paper's schedule) |
| `aggressive` | `very_aggressive` | 0 | dose 5.0 |

The dose axis is the V3 late-stage multiplier from `get_v3_ablation_params`.

**Scene set.** nuScenes val scenes whose published pair has a *moving* adversary — a stationary agent is
frozen for the whole rollout by `disable_control_on_stationary='current_speed'` (threshold 0.5 m/s) and
cannot be controlled at all, so including one only dilutes the result. Ordered by the mined encounter
distance, the first eight form tier A (`d_min < 15 m`, i.e. the conflict geometry is close enough for the
ego's safety margin to move at all):

```
tier A : scene-0268 scene-0274 scene-0269 scene-0098 scene-0276 scene-0273 scene-0555 scene-0104
tier B : scene-0275 scene-0018 scene-0909 scene-0103 scene-0557 scene-0556 scene-0560 scene-0093
```

## Metrics

`scripts/score_rollout.py` recomputes, from the raw rollout state, what the released pipeline does not:

- `dev_adv` / `dev_ego` / `dev_others` — mean per-agent displacement against the reference rollout. This
  is the size of the intervention and how localised it stays.
- `min_clearance` — closest ego/adversary approach, disk footprints subtracted.
- `min_ttc` — range over closing speed, minimum over the rollout.
- `hbr` / `hbr_ego` — hard-braking rate, paper appendix B.2 eq. 29–32 (`a_lon < -3.0 m/s²`).

**How to read them.** A control claim needs all three of:

1. `dev_adv` for a guided arm clearly exceeds `dev_adv` for `baseline_s1` (the divergence you get for
   free by changing the seed);
2. `dev_others` stays at or below that floor — the intervention is localised, not a global reshuffle;
3. the ego's safety margin (`min_clearance`, `min_ttc`) moves in the intended direction and does so
   more than the seed replicates move it.

`baseline_s0b` bounds the run-to-run nondeterminism; anything at that level is not an effect.

## Commands

```bash
export NUSCENES_ROOT=/path/to/nuscenes
export TRAJDATA_CACHE_DIR=/path/to/trajdata_cache
export COUNTERSCENE_CHECKPOINT_DIR=exps/counterscene/run0
export COUNTERSCENE_CHECKPOINT_KEY=iter28000.ckpt

# gradient probe first -- one scene, confirms the guidance is not inert
COUNTERSCENE_DEBUG_GRAD=1 python ccdiff/examples/scene_editor.py \
  --results_root_dir /tmp/gradprobe --num_scenes_per_batch 1 \
  --dataset_path "$NUSCENES_ROOT" --env trajdata \
  --policy_ckpt_dir "$COUNTERSCENE_CHECKPOINT_DIR" \
  --policy_ckpt_key "$COUNTERSCENE_CHECKPOINT_KEY" \
  --eval_class CCDiff --editing_source conflict \
  --selected_vehicles_path data/counterscene_selected_vehicles.json \
  --registered_name trajdata_nusc_counterscene \
  --part_control --controllable_agent 2 \
  --num_simulation_steps 50 --n_step_action 5 --seed 0 \
  --v3_ablation_variant full 2>&1 | grep '^\[GRAD\]' | head

# the five arms
bash scripts/run_controllability.sh

# score each arm against the unguided baseline
python scripts/score_rollout.py \
  --hdf5 results/controllability/full/scene_edit_eval/data.hdf5 \
  --reference_hdf5 results/controllability/baseline/scene_edit_eval/data.hdf5 \
  --selected_vehicles data/counterscene_selected_vehicles.json \
  --output results/controllability/metrics_full.json
```

The arms are independent and each needs about 3 GB of GPU memory, so they can be run in parallel on a
24 GB card; `run_controllability.sh` runs them serially by default.

## Probing a recorded ego

For the use case where the ego is a fixed logged trajectory and only the surrounding agents may be
controlled, add `--replay_ego` (or `COUNTERSCENE_REPLAY_EGO=1` for the script). The ego then keeps its
logged positions and yaws after intervention while the world model drives everything else, so
`min_clearance` and `hbr_ego` describe the *recorded* policy's safety envelope rather than a model ego
that reacts to the perturbation. `adversary_only` already detaches the ego from the guidance gradient, so
nothing about the counterfactual objective changes.

Note that `dev_ego` is then ~0 by construction and stops being a useful diagnostic; use `dev_others` to
check locality instead.

## Results

<!-- filled in from the run described above -->
