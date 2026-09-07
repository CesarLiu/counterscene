# Copyright 2026 CounterScene authors
# SPDX-License-Identifier: Apache-2.0

"""Build tbsim guidance configs from published CounterScene targets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


_FULL_V3_PARAMS = {
    "early_mult": 0.2,
    "mid_start_mult": 0.2,
    "mid_end_mult": 1.5,
    "late_max_mult": 3.0,
    "early_end": 0.3,
    "mid_end": 0.7,
    "enable_adaptive": True,
    "enable_jerk": True,
    "enable_conflict_aware": True,
}


def get_v3_ablation_params(variant: str = "full") -> Dict[str, Any]:
    """Return the paper guidance parameters or one ablation preset."""
    params = deepcopy(_FULL_V3_PARAMS)
    if variant == "full":
        return params
    if variant in {"no_progressive", "constant"}:
        params.update(
            early_mult=1.0,
            mid_start_mult=1.0,
            mid_end_mult=1.0,
            late_max_mult=1.0,
        )
    elif variant == "no_conflict_aware":
        params["enable_conflict_aware"] = False
    elif variant == "no_adaptive":
        params["enable_adaptive"] = False
    elif variant == "no_jerk":
        params["enable_jerk"] = False
    elif variant == "minimal":
        params.update(
            early_mult=1.0,
            mid_start_mult=1.0,
            mid_end_mult=1.0,
            late_max_mult=1.0,
            enable_adaptive=False,
            enable_jerk=False,
            enable_conflict_aware=False,
        )
    elif variant == "conservative":
        params.update(early_mult=0.1, mid_end_mult=1.0, late_max_mult=2.0)
    elif variant == "very_conservative":
        params.update(early_mult=0.05, mid_end_mult=0.6, late_max_mult=1.2)
    elif variant == "very_aggressive":
        params.update(early_mult=0.5, mid_end_mult=2.5, late_max_mult=5.0)
    else:
        choices = (
            "full, no_progressive, constant, no_conflict_aware, no_adaptive, "
            "no_jerk, minimal, very_conservative, conservative, very_aggressive"
        )
        raise ValueError(f"unknown V3 ablation '{variant}'; choose one of: {choices}")
    return params


def build_v3_guidance(
    selection: Mapping[str, Any],
    ego_scene_idx: int,
    adv_scene_idx: int,
    *,
    ablation: str = "full",
    total_horizon: int = 50,
    adversary_only: bool = True,
    add_map_collision: bool = True,
    map_collision_weight: float = 2.0,
) -> list:
    """Build one scene's tbsim guidance list.

    ``ego_scene_idx``/``adv_scene_idx`` are the agents' rows *within the
    trajdata scene* -- the same space as the published ``ego_idx``/``adv_idx``.
    tbsim resolves ``guide_cfg.agents`` as ``cur_scene_inds[agents]``, so any
    other index space (e.g. positions within ``control_idx``) silently guides
    the wrong agents.

    Published weights use the sign convention of an earlier loss. V3 is a
    positive distance loss, so its optimizer weight must be positive.
    """
    required = {
        "conflict_point",
        "ego_arrival_time",
        "adv_arrival_time",
        "conflict_type",
        "danger_score",
    }
    missing = sorted(required.difference(selection))
    if missing:
        raise KeyError(f"selected-vehicle entry is missing: {', '.join(missing)}")

    params = {
        "conflict_point": selection["conflict_point"],
        "ego_arrival_time": selection["ego_arrival_time"],
        "adv_arrival_time": selection["adv_arrival_time"],
        "ego_idx": ego_scene_idx,
        "adv_idx": adv_scene_idx,
        "conflict_type": selection["conflict_type"],
        "sub_type": selection.get("sub_type"),
        "danger_score": selection["danger_score"],
        "total_horizon": int(total_horizon),
        "adversary_only": bool(adversary_only),
        **get_v3_ablation_params(ablation),
    }
    guidance = [
        {
            "name": "conflict_point_guidance_v3",
            "weight": abs(float(selection.get("guidance_weight", 1.0))),
            "params": params,
            "agents": [int(ego_scene_idx), int(adv_scene_idx)],
        }
    ]
    if add_map_collision:
        guidance.append(
            {
                "name": "map_collision",
                "weight": float(map_collision_weight),
                "params": {"num_points_lw": (10, 10), "decay_rate": 0.9},
                "agents": "all",
            }
        )
    return guidance
