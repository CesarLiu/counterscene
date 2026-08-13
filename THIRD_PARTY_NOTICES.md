# Third-party notices

CounterScene contains or depends on the following projects.

## CCDiff

The repository history and much of the base model/runtime integration are
adapted from [HenryLHH/CCDiff](https://github.com/HenryLHH/CCDiff), originally
copyright Cruise LLC and distributed under Apache License 2.0. The repository
root `LICENSE` contains that license.

## traffic-behavior-simulation (tbsim)

`third_party/tbsim` is a modified vendored subset of
[NVlabs/traffic-behavior-simulation](https://github.com/NVlabs/traffic-behavior-simulation).
It is governed by the NVIDIA Source Code License-NC, including its
non-commercial research/evaluation limitation. A complete license copy is at
`third_party/tbsim/LICENSE`.

## spline-planner (Pplan)

The bootstrap script installs revision
`c3d172917f337531209149546670503fa8ed1d3c` of
[NVlabs/spline-planner](https://github.com/NVlabs/spline-planner). It is not
vendored here and is governed by the NVIDIA Source Code License-NC.

## trajdata

The bootstrap script installs revision
`79ea54ce36d8b2b09b4bb4b43579ef2520352158` of the research-compatible
[AIasd/trajdata](https://github.com/AIasd/trajdata) fork. Upstream trajdata is
copyright NVIDIA and distributed under Apache License 2.0. Consult the checked
out dependency for its complete notices.

## nuScenes

nuScenes data are not distributed by this repository. Users must download them
and comply with the nuScenes dataset terms separately.
