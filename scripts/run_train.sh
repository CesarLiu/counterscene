#!/usr/bin/env bash
# Modified by the CounterScene authors, 2026.

set -euo pipefail

COUNTERSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset directory}"

# Mitigates CUDA allocator fragmentation ("reserved >> allocated" OOMs) observed
# on long single-GPU runs; see the batch_size comment in
# third_party/tbsim/tbsim/configs/trajdata_nusc_scene_config.py.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

python "${COUNTERSCENE_ROOT}/ccdiff/examples/train.py" \
  --dataset_path "${NUSCENES_ROOT}" \
  --config_name trajdata_nusc_counterscene \
  --name "${COUNTERSCENE_RUN_NAME:-counterscene}" \
  --output_dir "${COUNTERSCENE_OUTPUT_DIR:-experiments}"
