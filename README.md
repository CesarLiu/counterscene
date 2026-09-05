<div align="center">
<img src="assets/logo.png" width="180" alt="CounterScene logo" />

# CounterScene

### Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation

Bowen Jing<sup>1,*</sup>, Ruiyang Hao<sup>2,*</sup>, Weitao Zhou<sup>3,†</sup>, Haibao Yu<sup>1,4,†</sup>

<sup>1</sup> Tuojing Intelligence, <sup>2</sup> King's College London<br>
<sup>3</sup> Tsinghua University, <sup>4</sup> The University of Hong Kong

[![arXiv](https://img.shields.io/badge/arXiv-2603.21104-b31b1b)](https://arxiv.org/abs/2603.21104)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2603.21104)
[![License](https://img.shields.io/badge/license-mixed-orange)](#license)

</div>

## News

- **Aug. 2026:** 🎉 CounterScene has been accepted as an **Oral Presentation at the ECCV 2026 Workshop**.
- **Mar. 2026:** CounterScene is available on [arXiv](https://arxiv.org/abs/2603.21104).

## Overview

CounterScene generates realistic safety-critical driving scenarios through
counterfactual causal reasoning. Given a safe scene, it identifies the agent
whose behavior is causally maintaining safety and applies a minimal intervention
to that agent during diffusion-based closed-loop generation.

The framework combines a Conflict Interaction Graph (CIG), conflict-aware
counterfactual guidance, and a tbsim closed-loop simulator. Other agents remain
controlled by the learned world model and can react naturally to the
intervention, preserving coherent multi-agent behavior.

<div align="center">
<img src="assets/methodology.jpg" width="920" alt="CounterScene methodology" />
</div>

This repository includes:

- the CIG-enabled CounterScene world model built from CCDiff;
- staged spatial-temporal counterfactual guidance and its ablations;
- training, closed-loop evaluation, rendering, and result-parsing entry points;
- the customized tbsim runtime used by CounterScene;
- 90 published scene-local ego/adversary choices in
  `data/counterscene_selected_vehicles.json`;
- offline conflict mining -- adversary selection and guidance targets, per
  paper appendix A.2 (`ccdiff/counterscene/selection.py`,
  `scripts/build_selected_vehicles.py`);
- lightweight artifact, configuration, CIG, guidance, and selection tests.

The released artifact was produced by the authors' own mining code rather than by
the implementation here. The published vehicle pairs and all targets needed by the
evaluation pipeline are provided as a human-readable, schema-validated JSON
artifact, and remain the default input for reproducing the paper's numbers.

## Installation

### 1. System requirements

The reference environment was developed for:

- Linux x86-64;
- an NVIDIA GPU and a driver compatible with CUDA 11.7;
- Git and Conda/Miniconda;
- Python 3.8;
- PyTorch 1.13.1, torchvision 0.14.1, and torchtext 0.14.1.

CPU-only execution has not been validated and is not recommended for training
or closed-loop diffusion evaluation.

### 2. Clone the repository

```bash
git clone https://github.com/TuojingAI/CounterScene.git
cd CounterScene
```

### 3. Create the Conda environment

The provided environment file installs the reference PyTorch/CUDA stack and
the Python dependencies used by the project:

```bash
conda env create -f environment.yml
conda activate counterscene
```

Confirm that PyTorch can see the GPU:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The expected output contains `torch: 1.13.1` and `cuda: True`.

### 4. Install CounterScene and pinned third-party dependencies

Run the bootstrap script from the repository root:

```bash
bash scripts/bootstrap_third_party.sh
```

The script performs the following operations:

1. clones the research-compatible trajdata fork at revision
   `79ea54ce36d8b2b09b4bb4b43579ef2520352158`;
2. clones NVIDIA spline-planner at revision
   `c3d172917f337531209149546670503fa8ed1d3c`;
3. installs trajdata, spline-planner, the vendored tbsim runtime, and
   CounterScene in editable mode.

The resulting dependency directories are placed under `third_party/` and are
ignored by Git.

### 5. Prepare nuScenes

Download the nuScenes `v1.0-trainval` metadata, samples, sweeps, and maps. Point
`NUSCENES_ROOT` to the directory containing them. A typical layout is:

```text
/path/to/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-trainval/
```

Set the dataset and cache locations with absolute paths:

```bash
export NUSCENES_ROOT=/path/to/nuscenes
export TRAJDATA_CACHE_DIR=/path/to/trajdata_cache
mkdir -p "$TRAJDATA_CACHE_DIR"
```

Add these exports to your shell profile if you want them to persist across
sessions. The first dataset access may take additional time while trajdata
builds its cache.

### 6. Verify the installation

Check the main packages:

```bash
python -c "import ccdiff, tbsim, trajdata; print('CounterScene imports: OK')"
```

Run the repository tests:

```bash
python -m unittest discover -s tests -v
```

All tests should pass in the reference environment. The CIG tensor test and V3
gradient test require PyTorch and are skipped automatically in a lightweight
environment without it.

## Training

After setting `NUSCENES_ROOT`, start CounterScene training with:

```bash
bash scripts/run_train.sh
```

Optional output settings can be supplied through environment variables:

```bash
export COUNTERSCENE_RUN_NAME=counterscene
export COUNTERSCENE_OUTPUT_DIR=/path/to/experiments
bash scripts/run_train.sh
```

The registered CounterScene configuration is
`trajdata_nusc_counterscene`. The original `trajdata_nusc_ccdiff`
configuration remains available only for upstream CCDiff compatibility.

## Closed-loop evaluation

Evaluation requires a CounterScene checkpoint and the `config.json` saved in
the same training run. Do not use an original CCDiff checkpoint: it does not
contain the CIG soft gate and is shape-incompatible with CounterScene.

Set the required paths:

```bash
export NUSCENES_ROOT=/path/to/nuscenes
export TRAJDATA_CACHE_DIR=/path/to/trajdata_cache
export COUNTERSCENE_CHECKPOINT_DIR=/path/to/checkpoint_run
export COUNTERSCENE_CHECKPOINT_KEY=iterXXXXX.ckpt
export COUNTERSCENE_RESULTS_DIR=/path/to/results
```

Run the 5-second closed-loop evaluation:

```bash
bash scripts/run_eval.sh
```

By default, the script reads the checked-in selections from
`data/counterscene_selected_vehicles.json`, evaluates one scene per batch, uses
a 50-frame horizon at 10 Hz, and controls the published ego/adversary pair. An
alternate reviewed JSON can be supplied with:

```bash
export COUNTERSCENE_SELECTED_VEHICLES=/path/to/selected_vehicles.json
```

The horizon and replanning interval can also be changed:

```bash
export COUNTERSCENE_SIMULATION_STEPS=100
export COUNTERSCENE_ACTION_STEPS=5
bash scripts/run_eval.sh
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the artifact schema,
evaluation protocol, V3 ablations, and checkpoint compatibility notes.

### Regenerating selections

`scripts/build_selected_vehicles.py` re-derives ego/adversary pairs and guidance
targets from a trajdata cache, implementing the offline conflict mining of paper
appendix A.2 (equations 10-17):

```bash
python scripts/build_selected_vehicles.py \
  --cache_dir "$TRAJDATA_CACHE_DIR" --env_name nusc_trainval \
  --output selected_vehicles.json \
  --compare_to data/counterscene_selected_vehicles.json
```

Every constant in `SelectionConfig` is taken from the paper. The mined targets
are not bit-identical to the released artifact -- see
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for measured agreement and
for the agent-index caveat that governs whether a regenerated artifact can be
fed to `run_eval.sh`.

## Qualitative example

<div align="center">
<img src="assets/103.png" width="900" alt="CounterScene merging cut-in example" />
</div>

Quantitative comparisons, ablations, and additional qualitative examples are
available in the paper.

## Repository structure

```text
ccdiff/counterscene/       artifact validation, CIG features, V3 configuration, selection
ccdiff/models/             CCDiff world model with CounterScene CIG support
ccdiff/examples/           training, evaluation, and result parsing
data/                      published selected-vehicle artifact
docs/                      reproducibility documentation
scripts/                   installation, training, evaluation, and selection helpers
tests/                     focused smoke and unit tests
third_party/tbsim/         customized tbsim runtime
```

## License

This is a mixed-license repository. CounterScene and inherited CCDiff code at
the repository root are distributed under Apache-2.0 (`LICENSE`). The vendored
`third_party/tbsim` tree is governed by the NVIDIA Source Code License-NC and is
restricted to non-commercial research or evaluation. NVIDIA spline-planner is
under the same non-commercial license, while trajdata is Apache-2.0. Read
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistribution or use.

## Citation

If you find CounterScene useful, please cite:

```bibtex
@article{jing2026counterscene,
  title={CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation},
  author={Jing, Bowen and Hao, Ruiyang and Zhou, Weitao and Yu, Haibao},
  journal={arXiv preprint arXiv:2603.21104},
  year={2026}
}
```

## Acknowledgements

CounterScene builds on
[CCDiff](https://github.com/HenryLHH/CCDiff),
[traffic-behavior-simulation](https://github.com/NVlabs/traffic-behavior-simulation),
[trajdata](https://github.com/NVlabs/trajdata), and
[spline-planner](https://github.com/NVlabs/spline-planner).
