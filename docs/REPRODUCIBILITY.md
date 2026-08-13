# Reproducibility guide

## Pipeline

1. Train the CIG-enabled world model with
   `trajdata_nusc_counterscene`.
2. Load the published ego/adversary choices and precomputed guidance targets
   from `data/counterscene_selected_vehicles.json`.
3. Run closed-loop tbsim evaluation with V3 staged conflict guidance.
4. Aggregate standard displacement, off-road, hard-brake, and collision
   metrics from the rollout output.

The old `trajdata_nusc_ccdiff` registration has an eight-feature CCDiff edge
encoder and is retained only for upstream checkpoint compatibility. A checkpoint
must be loaded with the registration recorded in its own `config.json`.

The vehicle-selection algorithm is intentionally excluded. The public pipeline
does not attempt to reproduce that private step and has no fallback selector.

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
