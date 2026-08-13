#!/usr/bin/env bash
# Modified by the CounterScene authors, 2026.

set -euo pipefail

COUNTERSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset directory}"
: "${COUNTERSCENE_CHECKPOINT_DIR:?Set COUNTERSCENE_CHECKPOINT_DIR}"
: "${COUNTERSCENE_CHECKPOINT_KEY:?Set COUNTERSCENE_CHECKPOINT_KEY}"

python "${COUNTERSCENE_ROOT}/ccdiff/examples/scene_editor.py" \
  --results_root_dir "${COUNTERSCENE_RESULTS_DIR:-results}" \
  --num_scenes_per_batch 1 \
  --dataset_path "${NUSCENES_ROOT}" \
  --env trajdata \
  --policy_ckpt_dir "${COUNTERSCENE_CHECKPOINT_DIR}" \
  --policy_ckpt_key "${COUNTERSCENE_CHECKPOINT_KEY}" \
  --eval_class CCDiff \
  --editing_source conflict \
  --selected_vehicles_path "${COUNTERSCENE_SELECTED_VEHICLES:-${COUNTERSCENE_ROOT}/data/counterscene_selected_vehicles.json}" \
  --registered_name trajdata_nusc_counterscene \
  --part_control \
  --controllable_agent 2 \
  --num_simulation_steps "${COUNTERSCENE_SIMULATION_STEPS:-50}" \
  --n_step_action "${COUNTERSCENE_ACTION_STEPS:-5}" \
  --save_every_n_frames 5
