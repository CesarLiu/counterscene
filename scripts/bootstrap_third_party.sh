#!/usr/bin/env bash
# Copyright 2026 CounterScene authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

COUNTERSCENE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${COUNTERSCENE_ROOT}/third_party"
TRAJDATA_REVISION="79ea54ce36d8b2b09b4bb4b43579ef2520352158"
PPLAN_REVISION="c3d172917f337531209149546670503fa8ed1d3c"

if [[ ! -d "${THIRD_PARTY_ROOT}/trajdata/.git" ]]; then
  git clone https://github.com/AIasd/trajdata.git "${THIRD_PARTY_ROOT}/trajdata"
fi
git -C "${THIRD_PARTY_ROOT}/trajdata" checkout "${TRAJDATA_REVISION}"
python -m pip install -r "${THIRD_PARTY_ROOT}/trajdata/trajdata_requirements.txt"
python -m pip install -e "${THIRD_PARTY_ROOT}/trajdata"

if [[ ! -d "${THIRD_PARTY_ROOT}/Pplan/.git" ]]; then
  git clone https://github.com/NVlabs/spline-planner.git "${THIRD_PARTY_ROOT}/Pplan"
fi
git -C "${THIRD_PARTY_ROOT}/Pplan" checkout "${PPLAN_REVISION}"
python -m pip install -e "${THIRD_PARTY_ROOT}/Pplan"

python -m pip install -e "${THIRD_PARTY_ROOT}/tbsim"
python -m pip install -e "${COUNTERSCENE_ROOT}"
