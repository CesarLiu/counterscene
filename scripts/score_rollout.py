#!/usr/bin/env python
"""Safety-margin / controllability metrics from a CounterScene rollout HDF5.

Used by scripts/run_controllability.sh; see docs/CONTROLLABILITY.md.

The released pipeline only computes off-road, collision, comfort and
critical-failure metrics. The quantities needed to characterise how an
intervention shifts the ego's safety envelope -- minimum clearance, minimum
TTC, and hard-braking rate -- are not part of it, so they are recomputed here
from the raw per-step rollout state (centroid / yaw / extent).

Hard braking follows the paper's appendix B.2 definition (eq. 29-32):
    v_t     = (p_{t+1} - p_t) / dt
    a_t     = (v_{t+1} - v_t) / dt
    a_lon,t = ||a_t|| * cos(yaw_t)
    event   <=> a_lon,t < -3.0 m/s^2
    HBR_scene = #events / #valid acceleration timesteps, averaged over scenes.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import h5py
import numpy as np


DT = 0.1
HARD_BRAKE_THRESHOLD = -3.0  # m/s^2, paper eq. 31


def _finite_diff_kinematics(centroid: np.ndarray, yaw: np.ndarray):
    """Per-agent longitudinal acceleration by finite differences (eq. 29-30)."""
    velocity = np.diff(centroid, axis=1) / DT              # (A, T-1, 2)
    acceleration = np.diff(velocity, axis=1) / DT          # (A, T-2, 2)
    accel_mag = np.linalg.norm(acceleration, axis=-1)      # (A, T-2)
    a_lon = accel_mag * np.cos(yaw[:, : accel_mag.shape[1]])
    return velocity, acceleration, a_lon


def hard_braking_rate(centroid: np.ndarray, yaw: np.ndarray, valid: np.ndarray) -> float:
    """Scene-level HBR (eq. 32): fraction of valid accel steps below threshold."""
    _, _, a_lon = _finite_diff_kinematics(centroid, yaw)
    # a_t depends on p_t, p_{t+1}, p_{t+2}: all three must be valid, otherwise a
    # departing agent's jump to the origin registers as a huge deceleration.
    n = a_lon.shape[1]
    valid_accel = valid[:, :n] & valid[:, 1 : n + 1] & valid[:, 2 : n + 2]
    total = int(np.count_nonzero(valid_accel))
    if total == 0:
        return float("nan")
    events = int(np.count_nonzero((a_lon < HARD_BRAKE_THRESHOLD) & valid_accel))
    return events / total


def pair_safety(
    centroid: np.ndarray,
    extent: np.ndarray,
    valid: np.ndarray,
    ego_idx: int,
    adv_idx: int,
) -> Dict[str, float]:
    """Ego/adversary clearance and time-to-collision over the rollout."""
    both_valid = valid[ego_idx] & valid[adv_idx]
    if not np.any(both_valid):
        return {"min_center_distance": float("nan"), "min_clearance": float("nan"),
                "min_ttc": float("nan"), "closing_speed_at_min": float("nan")}

    delta = centroid[adv_idx] - centroid[ego_idx]                 # (T, 2)
    distance = np.linalg.norm(delta, axis=-1)
    distance = np.where(both_valid, distance, np.inf)

    # Disk approximation of the two footprints, consistent with the repo's own
    # disk-based collision metric; the paper's CR itself uses oriented boxes.
    radii = 0.5 * np.linalg.norm(extent[..., :2], axis=-1)        # (A, T)
    clearance = distance - (radii[ego_idx] + radii[adv_idx])
    clearance = np.where(both_valid, clearance, np.inf)

    # TTC along the closing direction: range / closing speed, when closing.
    relative_velocity = np.diff(delta, axis=0) / DT               # (T-1, 2)
    range_step = distance[:-1]
    unit = np.divide(
        delta[:-1], np.maximum(range_step, 1e-6)[:, None],
        out=np.zeros_like(delta[:-1]), where=range_step[:, None] > 0,
    )
    closing_speed = -np.sum(relative_velocity * unit, axis=-1)    # >0 means approaching
    # the relative velocity at t needs both t and t+1 valid for both agents
    step_valid = both_valid[:-1] & both_valid[1:] & np.isfinite(range_step) & (closing_speed > 1e-3)
    ttc = np.where(step_valid, range_step / np.maximum(closing_speed, 1e-6), np.inf)

    min_idx = int(np.argmin(distance))
    return {
        "min_center_distance": float(np.min(distance)),
        "min_clearance": float(np.min(clearance)),
        "min_ttc": float(np.min(ttc)) if np.any(np.isfinite(ttc)) else float("nan"),
        "closing_speed_at_min": float(closing_speed[min(min_idx, len(closing_speed) - 1)]),
    }


def load_scene(group) -> Dict[str, np.ndarray]:
    centroid = np.asarray(group["centroid"])          # (A, T, 2)
    yaw = np.asarray(group["yaw"])                    # (A, T)
    extent = np.asarray(group["extent"])              # (A, T, 3)
    # Agents that leave the scene are written as exactly [0, 0] rather than NaN;
    # real nuScenes world coordinates are hundreds of metres from the origin, so
    # treating an exact origin as invalid is safe and necessary -- otherwise a
    # departed agent reads as a ~1.7 km jump.
    finite = np.isfinite(centroid).all(axis=-1) & np.isfinite(yaw)
    at_origin = np.all(centroid == 0.0, axis=-1)
    valid = finite & ~at_origin
    return {"centroid": centroid, "yaw": yaw, "extent": extent, "valid": valid}


def scene_metrics(
    scene: Dict[str, np.ndarray],
    ego_idx: int,
    adv_idx: Optional[int],
    reference: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["hbr"] = hard_braking_rate(scene["centroid"], scene["yaw"], scene["valid"])
    out["hbr_ego"] = hard_braking_rate(
        scene["centroid"][ego_idx : ego_idx + 1],
        scene["yaw"][ego_idx : ego_idx + 1],
        scene["valid"][ego_idx : ego_idx + 1],
    )
    if adv_idx is not None and adv_idx < scene["centroid"].shape[0]:
        out.update(pair_safety(scene["centroid"], scene["extent"], scene["valid"], ego_idx, adv_idx))

    # Per-agent deviation against an unguided reference rollout: how large the
    # intervention is, and how localised it stays.
    if reference is not None:
        a = min(scene["centroid"].shape[0], reference["centroid"].shape[0])
        t = min(scene["centroid"].shape[1], reference["centroid"].shape[1])
        both = scene["valid"][:a, :t] & reference["valid"][:a, :t]
        deviation = np.linalg.norm(
            scene["centroid"][:a, :t] - reference["centroid"][:a, :t], axis=-1
        )
        per_agent = np.array([
            deviation[i][both[i]].mean() if np.any(both[i]) else np.nan
            for i in range(a)
        ])
        out["dev_ego"] = float(per_agent[ego_idx]) if ego_idx < a else float("nan")
        if adv_idx is not None and adv_idx < a:
            out["dev_adv"] = float(per_agent[adv_idx])
            others = [i for i in range(a) if i not in (ego_idx, adv_idx)]
            out["dev_others_mean"] = float(np.nanmean(per_agent[others])) if others else float("nan")
            out["dev_others_max"] = float(np.nanmax(per_agent[others])) if others else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hdf5", required=True, help="rollout data.hdf5 to score")
    ap.add_argument("--reference_hdf5", default=None,
                    help="unguided rollout to measure intervention magnitude against")
    ap.add_argument("--selected_vehicles", default="data/counterscene_selected_vehicles.json")
    ap.add_argument("--output", default=None, help="write per-scene metrics as JSON")
    args = ap.parse_args()

    with open(args.selected_vehicles, "r", encoding="utf-8") as handle:
        published = json.load(handle)["selected_vehicles"]

    reference_scenes = {}
    if args.reference_hdf5:
        with h5py.File(args.reference_hdf5, "r") as ref:
            for key in ref.keys():
                reference_scenes[key] = load_scene(ref[key])

    rows = {}
    with h5py.File(args.hdf5, "r") as handle:
        for key in handle.keys():
            scene = load_scene(handle[key])
            scene_name = key.rsplit("_", 1)[0]
            entry = published.get(scene_name, {})
            rows[key] = scene_metrics(
                scene,
                ego_idx=int(entry.get("ego_idx", 0)),
                adv_idx=entry.get("adv_idx"),
                reference=reference_scenes.get(key),
            )
            rows[key]["scene_name"] = scene_name
            rows[key]["conflict_type"] = entry.get("conflict_type")

    numeric = [k for k in next(iter(rows.values())) if k not in ("scene_name", "conflict_type")] if rows else []
    summary = {
        k: float(np.nanmean([r[k] for r in rows.values() if isinstance(r.get(k), float)]))
        for k in numeric
    }
    print(f"scenes scored: {len(rows)}")
    for k, v in summary.items():
        print(f"  {k:26s} {v:10.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"per_scene": rows, "summary": summary}, handle, indent=2)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
