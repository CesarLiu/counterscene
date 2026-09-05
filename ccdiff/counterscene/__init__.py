# Copyright 2026 CounterScene authors
# SPDX-License-Identifier: Apache-2.0

"""Public CounterScene components built on top of CCDiff and tbsim."""

from .artifacts import load_selected_vehicles, resolve_scene_key
from .guidance import build_v3_guidance, get_v3_ablation_params

# ``selection`` needs numpy; artifact loading and guidance construction do not.
# Expose its API lazily so the dependency-light entry points stay importable in
# environments without the numerical stack.
_SELECTION_EXPORTS = frozenset(
    {
        "ConflictCandidate",
        "SceneTracks",
        "SelectionConfig",
        "build_selected_vehicles",
        "build_selection_entry",
        "danger_score",
        "guidance_weight",
        "iter_conflict_candidates",
        "select_adversary",
    }
)

__all__ = sorted(
    _SELECTION_EXPORTS
    | {
        "build_v3_guidance",
        "get_v3_ablation_params",
        "load_selected_vehicles",
        "resolve_scene_key",
    }
)


def __getattr__(name):
    if name in _SELECTION_EXPORTS:
        from . import selection

        return getattr(selection, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
