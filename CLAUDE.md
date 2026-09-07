# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Heads-up on the parent directory

This repo lives inside `/home/Cesar/diffCurricula/`, which has its own `CLAUDE.md` describing a
*different* project (SAFE-SIM + a curriculum extension, package `tbsim` at that repo's root). That file
gets auto-loaded into context but does **not** describe this codebase. CounterScene is a separate git
repository; ignore the parent instructions when working here.

## What this repo is

CounterScene generates safety-critical driving scenarios by counterfactual intervention: given a safe
scene, it identifies the agent whose behavior is causally maintaining safety and perturbs only that agent
during diffusion-based closed-loop generation, while the remaining agents keep reacting through the
learned world model. It is the ECCV 2026 workshop release built on
[CCDiff](https://github.com/HenryLHH/CCDiff) + [tbsim](https://github.com/NVlabs/traffic-behavior-simulation) + trajdata.

Two source trees with different licenses and different roles:

- `ccdiff/` — repository-root code (Apache-2.0): the CounterScene world model, its configs, entry points,
  and the `ccdiff/counterscene/` package holding everything genuinely new in this release.
- `third_party/tbsim/` — a **vendored and modified** tbsim runtime (NVIDIA Source Code License-NC,
  non-commercial only). It is not pristine upstream: the config registry, the guidance dispatch table,
  and `tbsim/utils/guidance_loss_v3.py` (the V3 conflict loss itself) all live here. When tracing behavior,
  expect to cross the boundary in both directions.

`third_party/trajdata/` and `third_party/Pplan/` are cloned at pinned revisions by
`scripts/bootstrap_third_party.sh` and are gitignored — they will be absent in a fresh checkout.

## Commands

### Setup

```bash
conda env create -f environment.yml && conda activate counterscene   # python 3.8, torch 1.13.1/cu117
bash scripts/bootstrap_third_party.sh    # clones pinned trajdata + spline-planner, pip installs all 4 editable
export NUSCENES_ROOT=/path/to/nuscenes
export TRAJDATA_CACHE_DIR=/path/to/trajdata_cache   # optional; defaults to ~/.unified_data_cache
```

Without conda (plain venv), `bootstrap_third_party.sh`'s editable installs need two workarounds:
`pip<24.1` (pip ≥24.1 rejects `pytorch-lightning==1.8.3.post0`'s malformed `torch>=1.9.*` specifier), and
`l5kit==1.5.0` installed separately (pinned in `third_party/tbsim/requirements.txt`, not `setup.py`, but
imported unconditionally by several tbsim modules even on the trajdata-only path) followed by
re-pinning `numpy==1.23.4` (l5kit's installer downgrades it; the two packages' declared numpy pins
conflict, but 1.23.4 works fine at runtime for both).

### Tests

There is no pytest config, linter config, or CI; `unittest` plus a shell syntax check is the whole gate
(see `CONTRIBUTING.md`):

```bash
python -m unittest discover -s tests -v
bash -n scripts/*.sh
python -m unittest tests.test_guidance_config -v                     # a single module
python -m unittest tests.test_registry.CounterSceneRegistryTest.test_counter_scene_has_an_independent_model_registration
```

The tests are deliberately dependency-tiered: `test_artifacts` and `test_guidance_config` are pure Python;
`test_cig`, `test_guidance_v3` and most of `test_selection` self-skip without torch or numpy;
`test_registry` needs numpy and imports through `third_party/tbsim` by inserting it on `sys.path`. A bare
interpreter therefore reports a handful of passes, many skips and exactly 1 error (`test_registry`) —
that is not a code failure, it means torch/numpy are missing. Any *other* error in a bare interpreter
means a numpy import leaked into the pure-Python tier.

### Train / evaluate / parse

```bash
bash scripts/run_train.sh                                            # config_name trajdata_nusc_counterscene
COUNTERSCENE_CHECKPOINT_DIR=... COUNTERSCENE_CHECKPOINT_KEY=iterXXXXX.ckpt bash scripts/run_eval.sh
python ccdiff/examples/parse_scene_edit_results.py --results_dir <rollout dir> --eval_out_dir ...
```

Both shell scripts are thin wrappers over `ccdiff/examples/{train,scene_editor}.py`; anything not exposed
as a `COUNTERSCENE_*` env var (e.g. `--v3_ablation_variant`, `--render`,
`--legacy_joint_conflict_guidance`) requires invoking `scene_editor.py` directly with the same flag set.

## Architecture

### Config registration decides checkpoint compatibility

`--registered_name` → `EXP_CONFIG_REGISTRY` in `third_party/tbsim/tbsim/configs/registry.py` → an
`ExperimentConfig` whose `algo_config` is either `CCDiffConfig` or `CounterSceneConfig`
(`ccdiff/configs/algo_config.py`). `CounterSceneConfig` subclasses `CCDiffConfig` and changes the edge
feature layout (`neighbor_inds = [0,1,4,5,10,11,12,13]` over an edge tensor extended with TTI/d_int/v_int,
vs. upstream's 8 CCDiff features), sets `motion_dist = "cig_soft"`, and enables `use_conflict_soft_gate`.
Those choices change tensor shapes, so a checkpoint **must** be loaded with the registration recorded in
its own saved `config.json`. An upstream CCDiff checkpoint is intentionally incompatible; the
`trajdata_nusc_ccdiff` entry exists only for upstream compatibility, never as a fallback.

### The Conflict Interaction Graph path

Three files, in order:

1. `ccdiff/counterscene/cig.py:compute_cig_features` — from `(batch, agents, agents, time, ...)` relative
   positions/headings/speeds/extents, forward-projects pairwise motion over `horizon_steps`, and returns
   normalized time-to-interaction, interaction-distance and speed edge terms plus diagnostics.
2. `ccdiff/models/ccdiff.py` (~lines 610-695) — calls it when `motion_dist == 'cig_soft'` and appends
   `cig_extra` to the edge tensor, which is what makes the `neighbor_inds` indices above valid.
3. `ccdiff/models/scenetemporal.py` (~lines 395-470) — in CIG mode the hard `social_attn_radius` distance
   mask is replaced by a learned **soft gate**: `sigmoid(conflict_gate(edge_feat))` becomes a log-space
   additive attention bias, so conflict relevance is learned rather than thresholded by distance.

### The closed-loop counterfactual evaluation path

`ccdiff/examples/scene_editor.py:run_scene_editor` is the orchestrator. With
`--editing_source conflict --part_control`:

- Two policies are composed: `CCDiff` (the world model, from
  `third_party/tbsim/tbsim/evaluation/policy_composers.py`) and `CCDiffHierarchicalPolicy`, which builds
  `CCDiffHybridPolicyControl` (`ccdiff/policies/hierarchical.py`) — a part-control wrapper that replays
  ground truth until intervention, then rolls out model outputs for the whole scene.
- Per scene, `resolve_scene_key` maps the runtime trajdata scene to a published key (`scene-%04d` or the
  scene name), `control_idx` is set from the published `ego_idx`/`adv_idx`, and
  `set_guidance_dim(control_idx, n_step_action)` restricts the guidance perturbation to those agents via
  `PartialPerturbationGuidance` (`ccdiff/utils/guidance_loss.py`, wired in `ccdiff/models/base_models.py`).
- `build_v3_guidance` (`ccdiff/counterscene/guidance.py`) turns one published entry into a tbsim guidance
  list: a `conflict_point_guidance_v3` entry plus a `map_collision` regularizer over all agents. It is
  merged into `guidance_config` alongside any config/heuristic guidance.
- `conflict_point_guidance_v3` dispatches through `GUIDANCE_FUNC_MAP` in
  `third_party/tbsim/tbsim/utils/guidance_loss.py` to `ConflictPointGuidanceLossV3`
  (`guidance_loss_v3.py`), which combines a spatial distance term to the conflict point, a temporal
  arrival term, and a jerk smoothness term, scaled by a rollout-stage multiplier computed from
  `global_t / total_horizon`. It is also registered as scene-level guidance in that file's
  `is_scene_level` list — a new scene-level guidance name must be added there too.

Index convention worth internalizing: `guide_cfg.agents` is resolved by tbsim as
`cur_scene_inds[guide_cfg.agents]` (`third_party/tbsim/tbsim/utils/guidance_loss.py`), i.e. as indices into
the scene's **own agent rows** — the same space as the published `ego_idx`/`adv_idx`. So
`build_v3_guidance` must be given those scene rows directly, *not* their positions within `control_idx`.
The release originally passed `control_idx.index(...)`, which put the whole V3 gradient on whichever
agents happened to occupy rows 0/1 while `PartialPerturbationGuidance` only allowed the real
`control_idx` rows to move — the two sets are disjoint, so the guidance was inert. Note also that
`ConflictPointGuidanceLossV3` hardcodes `ego_idx=0, adv_idx=1` *after* masking, so the ordering of the
selected rows (ascending, since the mask is boolean) is what decides which agent is treated as the
adversary; the `ego_idx`/`adv_idx` params only feed an equality check.

### The selected-vehicle artifact, and the mining stage beside it

`data/counterscene_selected_vehicles.json` (`schema_version: 1`, 90 scenes) supplies the ego/adversary
pairs and every precomputed guidance target (`conflict_point`, arrival times,
`conflict_type`/`sub_type`, `danger_score`, `guidance_weight`, `total_horizon`).
`ccdiff/counterscene/artifacts.py:load_selected_vehicles` validates every field and rejects anything
malformed; JSON only, pickle support is refused by design — do not add a pickle loader.

`ccdiff/counterscene/selection.py` implements the paper's appendix A.2 offline conflict mining
(equations 10–17), driven by `scripts/build_selected_vehicles.py`. Every constant in `SelectionConfig`
comes from the paper — do not "tune" them without a citation. It is deliberately **not** wired into
`scene_editor.py`: the released artifact stays the default evaluation input, and the evaluation path
keeps no fallback selector (a scene without a pair is skipped, never synthesized).

Things worth knowing before touching it:

- Eq. 10 searches `(t_e, t_a)` **independently** — the closest encounter need not be simultaneous, which
  is why a published `conflict_point` can sit off both agents' sampled paths. Anyone "fixing" this into
  a same-time closest approach will silently break the targets.
- `ego_arrival_time`/`adv_arrival_time` are exactly `tau_e`/`tau_a` from that search.
- Intersection scores have no distance bound and the arrival gap is frequently 0, so ranking is
  dominated by the largest relative speed among coincident encounters — very sensitive to which agents
  the cache contains.
- Agent indices follow trajdata's `SimulationScene.agents`: `agent_presence[start_frame]` filtered by
  the dataset's `only_types` (`["vehicle"]` in the eval configs), order preserved, ego at index 0.
- Validation status lives in `docs/REPRODUCIBILITY.md`: eq. 17 is exact on all 90 published entries, and
  when the same adversary is chosen the conflict point matches to ~0.23 m with `(tau_e, tau_a)` exact in
  14/19. Reproducing the published *choices* needs a cache built from the pinned trajdata revision. Do
  not present regenerated artifacts as reproducing the paper's numbers.

## Conventions and gotchas

- `ccdiff/counterscene/__init__.py` imports `selection` lazily through a module `__getattr__`, because
  `selection` needs numpy while `artifacts`/`guidance` deliberately do not. Keep it that way: importing
  it eagerly breaks the pure-Python test tier.
- Keep new CounterScene logic in `ccdiff/counterscene/` rather than duplicating vendored tbsim behavior
  (`CONTRIBUTING.md`). Edits inside `third_party/tbsim/` inherit the NC license.
- `--part_control`, a selected-vehicle artifact, and `--num_scenes_per_batch 1` are mutually required;
  `run_scene_editor` raises if any is missing. Scenes with no published pair are skipped, not synthesized.
- Published `guidance_weight` uses an older loss's sign convention. V3 is a positive distance loss, so
  `build_v3_guidance` takes `abs(...)`; `tests/test_guidance_config.py` pins this.
- The release default is `adversary_only=True` (detached ego reference). `--legacy_joint_conflict_guidance`
  restores the older joint-gradient behavior; both must keep working.
- `--v3_ablation_variant` presets live in one place, `get_v3_ablation_params`; add ablations there and to
  `scene_editor.py`'s `choices` list together.
- Workshop protocol defaults (nuScenes val scenes 0..99, 3 s history, start frame `history_num_frames+1`,
  50-frame/5 s horizon, `n_step_action` 5, seed 0, 1 sim per scene) are baked into `SceneEditingConfig`
  and `scripts/run_eval.sh`; `docs/REPRODUCIBILITY.md` is the authority on the protocol, the artifact
  schema, and checkpoint compatibility.
- Never commit dataset files, checkpoints, raw rollout outputs, machine-specific absolute paths, or
  private selection intermediates.
- Both `train.py --registered_name/--config_name trajdata_nusc_counterscene` and
  `scene_editor.py --editing_source conflict --part_control` were unconditionally broken on a clean
  checkout (dispatch-name bug, missing `import copy`, `constraint_config=None`), plus a console-only bug
  that made `conflict_point_guidance_v3` always print `nan` on every `--part_control` run regardless of
  whether guidance was actually working. All four are fixed now; see "Environment verification and
  pipeline fixes" in `docs/REPRODUCIBILITY.md` before assuming a similar crash/NaN print is expected.
