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

**How to read them.** Start with `baseline_s0b`: it bounds run-to-run nondeterminism, and in practice it
is exactly zero (see Results), which means every guided-vs-`baseline` difference is caused by the
guidance and no significance threshold is needed. Confirm that first — if it is *not* zero on your setup,
it becomes the floor everything else must clear. Then a control claim needs:

1. `dev_adv` well above that floor — the adversary actually moved;
2. `dev_others` much smaller than `dev_adv` — the intervention is localised rather than a global
   reshuffle (it will not be zero; the other agents legitimately react through the world model);
3. the ego's safety margin (`min_clearance`, `min_ttc`) moving in the intended direction.

`baseline_s1` is a *different* seed. Its deviation is chaotic divergence, not measurement noise, so it is
a useful sense of scale ("how much does the scene move on its own?") but not the floor.

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

`run_controllability.sh` runs the arms serially. Do not parallelise all five on one card: per-process
memory grows with the current scene's agent count, so they fit while the early scenes are small and then
OOM at different scenes, leaving each arm with a different prefix of the set. Two concurrent arms on a
24 GB card is safe.

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

Checkpoint `iter28000` (a partially trained model — 28 k of the scheduled 33 k steps), seven tier-A
scenes, 50 simulation steps. Every figure is a mean over those scenes; `dev_*` is mean per-agent
displacement against the `baseline` rollout.

| arm | dose | dev_adv | dev_ego | dev_others | min_clearance | min_ttc |
|---|---|---|---|---|---|---|
| `baseline` | 0.0 | — | — | — | 11.67 | 11.26 |
| `baseline_s0b` | 0.0 | **0.000** | **0.000** | **0.000** | 11.67 | 11.26 |
| `baseline_s1` | 0.0 | 2.04 | 2.45 | 0.67 | 9.35 | 3.97 |
| `conservative` | 1.2 | 6.73 | 4.10 | 0.71 | 9.33 | 8.34 |
| `full` | 3.0 | 6.28 | 3.29 | 0.75 | 9.70 | 5.50 |
| `aggressive` | 5.0 | 6.05 | 3.66 | 0.74 | 9.34 | 5.38 |

**The rollout is bit-deterministic at a fixed seed.** `baseline_s0b` reproduces `baseline` exactly —
every deviation is 0.000 and every safety metric is identical. There is therefore no noise floor to clear
and no statistical argument to make: the whole difference between a guided arm and `baseline` is caused
by the guidance. (`baseline_s1` is a *different* seed; its 2.04 m is chaotic divergence, not measurement
noise, and it is the wrong yardstick now that the same-seed floor is known to be zero.)

**The intervention is real, directional and localised.** The adversary moves ~6.3 m on average — three
times what changing the seed entirely does — while the uncontrolled agents move 0.75 m, about an eighth
as far. Those 0.75 m are genuine world-model propagation, not noise. The ego's margin closes
consistently: `min_clearance` 11.67 → ~9.4 m and `min_ttc` 11.26 → 5.4–8.3 s.

**It is not dose-controllable through the ablation variants.** `dev_adv` is flat (6.73 / 6.28 / 6.05) and
if anything decreases with dose. The cause is mechanical, not statistical: the guidance takes exactly one
Adam step per denoising step (`grad_steps: 1`), and Adam's first step is `lr * sign(grad)` — independent
of gradient magnitude:

```
stage multiplier  0.2  ->  step = [-0.3, 0.3, -0.3, -0.3]
stage multiplier  5.0  ->  step = [-0.3, 0.3, -0.3, -0.3]
```

So the V3 stage multiplier, the `guidance_weight` in the artifact, and every `--v3_ablation_variant`
preset change only the *direction* of the perturbation, never its size. **The knob that actually scales
the intervention is `guidance_optimization_params.lr`** (and secondarily `grad_steps`), not any loss
weight. Sweep that instead when a dose axis is needed.

### Caveat on this particular run

The five arms were launched in parallel on one 24 GB card and OOMed part-way through the set, at
different scenes: they cover 9 / 7 / 15 / 16 / 15 / 13 of the 16 scenes. The table above is restricted to
the seven scenes every arm completed, which happen to be seven of the eight tier-A scenes. The
conclusions rest on those seven; the tier-B breadth check has not been run. `run_controllability.sh` now
runs the arms serially and reports a non-zero exit instead of hiding it.
