# Copyright 2026 CounterScene authors
# SPDX-License-Identifier: Apache-2.0

"""Validation and lookup helpers for published CounterScene selections.

This module intentionally contains no vehicle-selection implementation. It only
loads the scene-local vehicle pairs and guidance targets released with the code.
"""

from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


PathLike = Union[str, Path]
SelectedVehicles = Dict[str, Dict[str, Any]]

_REQUIRED_FIELDS = {
    "scene_name",
    "ego_idx",
    "adv_idx",
    "conflict_point",
    "conflict_type",
    "ego_arrival_time",
    "adv_arrival_time",
    "danger_score",
    "guidance_weight",
    "total_horizon",
}
_CONFLICT_TYPES = {"intersection", "following"}


def _validate_entry(scene_name: str, entry: Mapping[str, Any]) -> Dict[str, Any]:
    missing = sorted(_REQUIRED_FIELDS.difference(entry))
    if missing:
        raise ValueError(f"{scene_name}: missing fields: {', '.join(missing)}")
    if entry["scene_name"] != scene_name:
        raise ValueError(f"{scene_name}: scene_name does not match its mapping key")

    for field in ("ego_idx", "adv_idx", "ego_arrival_time", "adv_arrival_time"):
        if not isinstance(entry[field], int) or isinstance(entry[field], bool):
            raise TypeError(f"{scene_name}: {field} must be an integer")
    if entry["ego_idx"] < 0 or entry["adv_idx"] < 0:
        raise ValueError(f"{scene_name}: vehicle indices must be non-negative")
    if entry["ego_idx"] == entry["adv_idx"]:
        raise ValueError(f"{scene_name}: ego_idx and adv_idx must be different")

    point = entry["conflict_point"]
    if not isinstance(point, list) or len(point) != 2 or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in point
    ):
        raise TypeError(f"{scene_name}: conflict_point must be a two-number list")
    if entry["conflict_type"] not in _CONFLICT_TYPES:
        raise ValueError(
            f"{scene_name}: conflict_type must be one of {sorted(_CONFLICT_TYPES)}"
        )
    for field in ("danger_score", "guidance_weight"):
        if (
            not isinstance(entry[field], (int, float))
            or isinstance(entry[field], bool)
            or not math.isfinite(entry[field])
        ):
            raise TypeError(f"{scene_name}: {field} must be numeric")
    if not isinstance(entry["total_horizon"], int) or entry["total_horizon"] <= 0:
        raise ValueError(f"{scene_name}: total_horizon must be a positive integer")
    for field in ("ego_arrival_time", "adv_arrival_time"):
        if not 0 <= entry[field] < entry["total_horizon"]:
            raise ValueError(
                f"{scene_name}: {field} must lie within total_horizon"
            )

    return dict(entry)


def load_selected_vehicles(path: PathLike) -> SelectedVehicles:
    """Load and validate the public, JSON-only vehicle-selection artifact."""
    artifact_path = Path(path)
    if artifact_path.suffix.lower() != ".json":
        raise ValueError("selected vehicle artifacts must use the JSON format")
    with artifact_path.open("r", encoding="utf-8") as artifact_file:
        payload = json.load(artifact_file)

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("selected vehicle artifact must have schema_version 1")
    entries = payload.get("selected_vehicles")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("selected_vehicles must be a non-empty mapping")
    validated = {}
    for scene_name, entry in entries.items():
        if not isinstance(scene_name, str) or not isinstance(entry, dict):
            raise TypeError("each selected_vehicles item must map a string to an object")
        validated[scene_name] = _validate_entry(scene_name, entry)
    return validated


def resolve_scene_key(
    scene_index: Any,
    current_name: Optional[str],
    selected_vehicles: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Resolve a runtime trajdata scene to a published selection key."""
    candidates = [current_name]
    if isinstance(scene_index, Integral):
        candidates.append(f"scene-{int(scene_index):04d}")
    candidates.extend([scene_index, str(scene_index)])
    for candidate in candidates:
        if candidate is not None and candidate in selected_vehicles:
            return str(candidate)
    return None
