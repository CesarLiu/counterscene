#!/usr/bin/env bash
# Modified by the CounterScene authors, 2026.

set -euo pipefail

COUNTERSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset directory}"

python "${COUNTERSCENE_ROOT}/ccdiff/examples/train.py" \
  --dataset_path "${NUSCENES_ROOT}" \
  --config_name trajdata_nusc_counterscene \
  --name "${COUNTERSCENE_RUN_NAME:-counterscene}" \
  --output_dir "${COUNTERSCENE_OUTPUT_DIR:-experiments}"
