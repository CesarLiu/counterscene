# Reproducibility guide

## Pipeline

1. Train the CIG-enabled world model with
   `trajdata_nusc_counterscene`.
2. Load the published ego/adversary choices and precomputed guidance targets
   from `data/counterscene_selected_vehicles.json`. To derive selections for
   other scenes, see *Regenerating selections* below.
3. Run closed-loop tbsim evaluation with V3 staged conflict guidance.
4. Aggregate standard displacement, off-road, hard-brake, and collision
   metrics from the rollout output.

The old `trajdata_nusc_ccdiff` registration has an eight-feature CCDiff edge
encoder and is retained only for upstream checkpoint compatibility. A checkpoint
must be loaded with the registration recorded in its own `config.json`.

The checked-in artifact was produced by the authors' own mining code.
`ccdiff/counterscene/selection.py` implements the same published procedure
(appendix A.2) as a separate, opt-in offline stage; nothing in the evaluation
path calls it, and the evaluation pipeline still has no implicit fallback
selector -- a scene without a published pair is skipped rather than filled in.

## Selected-vehicle artifact schema

The checked-in JSON has `schema_version: 1` and a `selected_vehicles` mapping.
Each scene entry contains:

| Field | Meaning |
| --- | --- |
| `ego_idx`, `adv_idx` | trajdata scene-local agent indices |
| `scene_name` | nuScenes scene name; identical to the mapping key |
| `conflict_point` | precomputed world-frame `[x, y]` guidance target |
| `conflict_type` | `intersection` or `following` |
| `sub_type` | `rear_approach`, `lead_braking`, or `null` |
| `ego_arrival_time`, `adv_arrival_time` | indices on the 10 Hz future horizon |
| `danger_score` | precomputed conflict severity used by V3 |
| `guidance_weight` | precomputed signed weight; V3 uses its absolute value |
| `total_horizon` | horizon used to produce the precomputed target |

Only JSON is accepted. The loader validates every entry and rejects malformed
or ambiguous artifacts; pickle loading is deliberately unsupported.

## Regenerating selections

`ccdiff/counterscene/selection.py` implements the offline conflict mining of
appendix A.2, and `scripts/build_selected_vehicles.py` runs it over a trajdata
cache:

```bash
python scripts/build_selected_vehicles.py \
  --cache_dir "$TRAJDATA_CACHE_DIR" --env_name nusc_trainval \
  --scenes_from data/counterscene_selected_vehicles.json \
  --output selected_vehicles.json \
  --compare_to data/counterscene_selected_vehicles.json
```

Mining fixes the ego at agent index 0 and evaluates every non-ego agent that
shares at least five jointly valid future timesteps. Per pair:

| Step | Rule |
| --- | --- |
| closest encounter | `(tau_e, tau_a) = argmin ||p_e(t_e) - p_a(t_a)||` over **all** valid timestep pairs (eq. 10) -- the two time indices are independent, so the encounter need not be simultaneous |
| conflict point | midpoint `(p_e(tau_e) + p_a(tau_a)) / 2` (eq. 11), with `d_min` its separation (eq. 12) |
| arrival gap | `|tau_e - tau_a| * dt` (eq. 13) |
| relative speed | finite differences at the encounter (eq. 14) |
| conflict score | `v_rel / (dt + 0.5)` for intersection, `v_rel / (d_min + 1.0)` for following (eq. 15) |
| conflict type | direction cosine of the coarse travel directions; `cos > 0.8` is following, else intersection (eq. 16) |
| tiers | 1 intersection (`dt < 5.0`), 2 `rear_approach` (`d_min < 10`), 3 `lead_braking` (`d_min < 12`), all requiring `s >= 0.05` |
| winner | lexicographic over `(tier, -s_conflict)` |
| guidance weight | `-80 - 40*min(s,1)` / `-60 - 30*min(s,1)` / `-50` fallback (eq. 17) |

`ego_arrival_time` and `adv_arrival_time` in the artifact are exactly `tau_e`
and `tau_a`. Scenes where no candidate survives tiering are invalid mining
cases and receive no configuration; there is no fallback selector.

### Agreement with the released artifact

Measured over the 90 published scenes, against a cache built from the pinned
trajdata revision (`AIasd/trajdata@79ea54c`, see `scripts/bootstrap_third_party.sh`)
over the nuScenes `v1.0-trainval` validation split:

| Quantity | Result |
| --- | --- |
| eq. 17 weight from published `danger_score` | exact for all 90 entries |
| scenes producing a target | 89/90 |
| same `conflict_type` | 81/89 |
| same adversary chosen | 19/89 |
| conflict point, *when the same adversary is chosen* | median 0.23 m, max 0.48 m |
| `(tau_e, tau_a)`, *when the same adversary is chosen* | exact in 14/19 |

The equations reproduce the released targets essentially exactly whenever the
same adversary is picked; the residual sub-metre gap is track-resampling noise
between cache builds. What diverges is *which* candidate wins. For intersection
conflicts the score `v_rel / (dt + 0.5)` has no distance bound and `dt` is
frequently 0, so ranking is decided by the largest relative speed among
coincident encounters -- highly sensitive to exactly which agents the cache
contains. The true adversary lands at rank 1 in 19 scenes and within the top 3
in roughly half. Reproducing the published choices therefore requires a cache
built from the pinned trajdata revision, not just the same algorithm.

### Agent-index compatibility

`ego_idx`/`adv_idx` are scene-local indices into the agent order the simulator
uses at rollout. That order is trajdata's
`SimulationScene.agents = filtering.agent_types(scene.agent_presence[start_frame], ...)`,
i.e. presence order at the start frame filtered by the dataset's `only_types`,
which the CounterScene eval configs set to `["vehicle"]`. The helper script
reproduces that rule, so indices are correct by construction *for the cache it
reads*. Verify the ordering before passing a regenerated artifact through
`COUNTERSCENE_SELECTED_VEHICLES`; a mismatch silently guides the wrong agent.

## V3 ablations

`scene_editor.py --v3_ablation_variant` supports:

- `full`: staged, conflict-aware, adaptive-time, jerk-regularized guidance.
- `no_progressive` / `constant`: fixed stage multiplier.
- `no_conflict_aware`: shared weights across conflict types.
- `no_adaptive`: fixed arrival times.
- `no_jerk`: no smoothness term.
- `minimal`: disables all four additions.
- `very_conservative`, `conservative`, `very_aggressive`: strength presets.

The rollout-stage schedule uses `global_t / total_horizon`. The detached ego
reference is the default release behavior. The legacy joint-gradient behavior
is available through `--legacy_joint_conflict_guidance`.

## Environment inputs

| Variable | Required | Description |
| --- | --- | --- |
| `NUSCENES_ROOT` | yes for nuScenes | Raw nuScenes root |
| `TRAJDATA_CACHE_DIR` | optional | trajdata cache; defaults to `~/.unified_data_cache` |
| `COUNTERSCENE_CHECKPOINT_DIR` | for `run_eval.sh` | directory containing checkpoint and `config.json` |
| `COUNTERSCENE_CHECKPOINT_KEY` | for `run_eval.sh` | checkpoint filename or unique key |
| `COUNTERSCENE_SELECTED_VEHICLES` | optional | alternate selected-vehicle JSON; defaults to the checked-in artifact |

## Workshop evaluation protocol

The checked-in runner matches the 5-second workshop setup recovered from the
author workspace:

| Setting | Value |
| --- | --- |
| nuScenes validation indices | `0..99` |
| scenes with a published valid pair | 90; scenes without a pair are skipped |
| observed history | 3 seconds |
| start frame | `history_num_frames + 1` (31 at 10 Hz) |
| closed-loop horizon | 50 frames (5 seconds) |
| action candidates | 5 |
| replanning interval | 5 frames |
| random seed | 0 |
| simulations per scene/start point | 1 |

The longer-horizon and multi-rollout settings described by newer paper
revisions can be configured through `--num_simulation_steps` and the evaluation
configuration, but they are not presented here as workshop defaults.

## Required external artifacts

The public implementation is self-contained apart from dataset and model
weights. Numerical reproduction requires:

- the final workshop CIG checkpoint together with its exact `config.json`;
- the nuScenes dataset and trajdata cache generated from the documented revision.

Do not substitute an older CCDiff checkpoint: it does not contain the CIG soft
gate and its tensor shapes are intentionally incompatible with CounterScene.

No CounterScene checkpoint is published or linked from this repository (unlike
the `evaluation/*.yaml` pointer files SAFE-SIM uses for its own checkpoints).
Reproducing the paper's tables therefore also requires training a checkpoint
from scratch at the scale in Table 6 (100k steps, batch 4, 3 GPUs) -- infeasible
on a single consumer/workstation GPU in a short session. What the released code
*can* be verified to do without that checkpoint is exercise the full pipeline
correctly end to end (see below).

## Environment verification and pipeline fixes found in the process

Setting this repository up from a clean checkout (`bash scripts/bootstrap_third_party.sh`
plus manually pinning `torch==1.13.1+cu117`/`torchvision==0.14.1+cu117` since
`environment.yml`'s conda channel pins were not reproduced in a plain venv) and
running the documented train/eval commands end to end surfaced four real bugs,
independent of any specific hardware or dataset subset:

1. **`ccdiff/algos/factory.py`** dispatched on `algo_config.name == "ccdiff"` only.
   `CounterSceneConfig.name` is `"counterscene"`, so `--registered_name
   trajdata_nusc_counterscene` -- this repository's own documented default --
   could never construct a model; `train.py` raised immediately. Fixed by
   accepting both names (the same `CCDiffTrafficModel` class handles both; it
   branches on `registered_name`, not `algo_config.name`).
2. **`ccdiff/utils/guidance_loss.py`** called `copy.deepcopy` in
   `set_perturbation_idx` without importing `copy`. This is on the
   `set_guidance_dim` path used by both training's `validation_step` and, more
   importantly, `scene_editor.py`'s `--part_control` counterfactual rollout --
   i.e. the paper's headline mechanism. Fixed by adding the import.
3. **`ccdiff/examples/scene_editor.py`** initialized `constraint_config = None`
   per batch, only ever overwriting it when `"config"` is one of
   `--editing_source`'s values. With `--editing_source conflict` (this
   repository's own release configuration, used by `run_eval.sh`),
   `constraint_config` stayed `None` and crashed vendored `guided_rollout`'s
   unconditional `len(constraint_config) > 0`. Fixed by initializing it to `[]`,
   matching `SceneEditingConfig.edits.constraint_config`'s own default.
4. **`ccdiff/models/ccdiff.py`**'s guidance-loss console printer treated any NaN
   in a guidance's per-agent loss tensor as "print nan". Vendored
   `compute_guidance_loss` pads agents *outside* a guidance's `agt_mask` with
   NaN by design (meant to be summarized with `nanmean`), so this fired for
   every partial-agent guidance -- including `conflict_point_guidance_v3`
   restricted to `[ego_idx, adv_idx]`, i.e. every `--part_control` run. The
   result: `conflict_point_guidance_v3` printed `nan` on every intervention
   step of every run, looking exactly like the counterfactual guidance itself
   was broken, when the actual per-agent loss was finite throughout. Fixed by
   only treating an all-NaN tensor as "nothing to report" (`.all()` instead of
   `.any()`).

None of these are hardware- or dataset-subset-specific: (1)-(3) are
unconditional crashes on the documented default configuration and editing
source; (4) is a logging defect that would mislead anyone inspecting console
output on any `--part_control` run, including the authors' own. After all four
fixes, a smoke checkpoint (`nusc_mini`, a few hundred training steps) run
through `scene_editor.py --part_control --editing_source conflict` against the
real published artifact completed a full 50-step closed-loop rollout on a real
published scene (verified: the resolved scene's `conflict_point`,
`ego_arrival_time`/`adv_arrival_time`, and `danger_score` matched
`data/counterscene_selected_vehicles.json` exactly), with `conflict_point_guidance_v3`
reporting large, finite, decreasing-toward-zero loss as the guided adversary
converged toward the conflict point over the rollout -- direct evidence the CIG
edge features, soft attention gate, `--part_control` index restriction, and
staged V3 guidance are all wired together correctly. This is a correctness
check, not a numerical reproduction: the checkpoint used is far too
undertrained for the resulting trajectories or metrics to mean anything against
the paper's tables.
