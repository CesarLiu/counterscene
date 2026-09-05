#!/usr/bin/env python
# Copyright 2026 CounterScene authors
# SPDX-License-Identifier: Apache-2.0

"""Run offline conflict mining over a trajdata unified cache.

Implements the mining protocol of CounterScene appendix A.2 on top of
``ccdiff/counterscene/selection.py``. It reads trajdata's cached agent tracks
directly -- ``<cache>/<env>/<scene>/agent_data_dt0.10.feather`` plus the scene
metadata dill -- so it does not need the full simulator stack to run.

    python scripts/build_selected_vehicles.py \
        --cache_dir ~/.unified_data_cache --env_name nusc_trainval \
        --scenes_from data/counterscene_selected_vehicles.json \
        --output /tmp/selected_vehicles.json --compare_to data/counterscene_selected_vehicles.json

Agent indexing follows the rollout convention, as A.2 requires: trajdata's
``SimulationScene`` takes ``scene.agent_presence[start_frame]`` filtered by the
dataset's ``only_types`` (``["vehicle"]`` for the CounterScene eval configs) and
preserves that order, with the ego at index 0. This script reproduces that rule
from the cache. Indices are therefore only as faithful as the cache: one built
with a different trajdata revision can enumerate a different agent set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccdiff.counterscene.artifacts import load_selected_vehicles  # noqa: E402
from ccdiff.counterscene.selection import (  # noqa: E402
    SceneTracks,
    SelectionConfig,
    build_selected_vehicles,
    build_selection_entry,
    select_adversary,
)


DEFAULT_CACHE = os.environ.get("TRAJDATA_CACHE_DIR", "~/.unified_data_cache")


def _load_scene(cache_dir: Path, scene_name: str, dt: float = 0.1):
    """Return (scene metadata, agent dataframe) from the trajdata cache."""
    import dill
    import pandas as pd

    scene_dir = cache_dir / scene_name
    metadata_path = scene_dir / f"scene_metadata_dt{dt:.2f}.dill"
    agent_path = scene_dir / f"agent_data_dt{dt:.2f}.feather"
    if not metadata_path.exists() or not agent_path.exists():
        raise FileNotFoundError(f"{scene_name}: no cached tracks at dt={dt:.2f} in {scene_dir}")
    with metadata_path.open("rb") as handle:
        metadata = dill.load(handle)
    return metadata, pd.read_feather(agent_path)


def _agent_order(metadata, start_frame: int, only_types: Optional[Sequence[str]]) -> List:
    """Reproduce ``SimulationScene``'s agent list: presence order, type filter."""
    if start_frame >= len(metadata.agent_presence):
        return []
    present = list(metadata.agent_presence[start_frame])
    if only_types:
        wanted = {name.upper() for name in only_types}
        present = [
            agent for agent in present if str(agent.type).split(".")[-1].upper() in wanted
        ]
    return present


def scene_tracks(
    metadata,
    frame: "object",
    start_frame: int,
    horizon: int,
    only_types: Optional[Sequence[str]],
) -> Optional[SceneTracks]:
    """Slice one scene's cached ground-truth futures into a :class:`SceneTracks`."""
    agents = _agent_order(metadata, start_frame, only_types)
    if len(agents) < 2 or agents[0].name != "ego":
        return None

    steps = np.arange(start_frame, start_frame + horizon)
    indexed = frame.set_index(["agent_id", "scene_ts"]).sort_index()

    positions = np.zeros((len(agents), horizon, 2), dtype=float)
    availability = np.zeros((len(agents), horizon), dtype=bool)

    for row, agent in enumerate(agents):
        try:
            track = indexed.loc[agent.name]
        except KeyError:
            continue
        track = track.reindex(steps)
        availability[row] = track["x"].notna().to_numpy()
        positions[row] = track[["x", "y"]].fillna(0.0).to_numpy(dtype=float)

    return SceneTracks(
        positions=positions,
        availability=availability,
        agent_names=[agent.name for agent in agents],
    )


def _scene_names(args, cache_dir: Path) -> List[str]:
    if args.scenes:
        return list(args.scenes)
    if args.scenes_from:
        return sorted(load_selected_vehicles(args.scenes_from))
    names = sorted(
        entry.name
        for entry in cache_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("scene-")
    )
    return names[: args.limit] if args.limit else names


def _report_comparison(generated: Dict[str, dict], reference_path: str) -> None:
    reference = load_selected_vehicles(reference_path)
    shared = sorted(set(generated) & set(reference))
    if not shared:
        print("\nno overlapping scenes with the reference artifact")
        return
    same_pair = sum(
        1 for name in shared if generated[name]["adv_idx"] == reference[name]["adv_idx"]
    )
    same_type = sum(
        1
        for name in shared
        if generated[name]["conflict_type"] == reference[name]["conflict_type"]
    )
    point_gap = [
        float(
            np.linalg.norm(
                np.array(generated[name]["conflict_point"])
                - np.array(reference[name]["conflict_point"])
            )
        )
        for name in shared
    ]
    print(f"\ncomparison against {reference_path} over {len(shared)} shared scenes")
    print(f"  same adversary index : {same_pair}/{len(shared)}")
    print(f"  same conflict type   : {same_type}/{len(shared)}")
    print(f"  conflict-point gap   : median {np.median(point_gap):.1f} m")
    print(
        "  note: eq. 10-17 are implemented as published; residual disagreement is\n"
        "  dominated by which candidate wins, which is sensitive to the cached\n"
        "  agent set (see docs/REPRODUCIBILITY.md)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE, help="trajdata unified cache root")
    parser.add_argument("--env_name", default="nusc_trainval", help="cached environment name")
    parser.add_argument("--scenes", nargs="+", default=None, help="explicit scene names")
    parser.add_argument("--scenes_from", default=None, help="take scene names from this artifact")
    parser.add_argument("--limit", type=int, default=None, help="cap on scanned scenes")
    parser.add_argument("--start_frame", type=int, default=31, help="closed-loop start frame")
    parser.add_argument("--total_horizon", type=int, default=50, help="guidance horizon in frames")
    parser.add_argument("--dt", type=float, default=0.1, help="cached step length in seconds")
    parser.add_argument(
        "--only_types",
        nargs="*",
        default=["vehicle"],
        help="agent types the simulator keeps; matches the eval config's only_types",
    )
    parser.add_argument("--output", required=True, help="where to write the JSON artifact")
    parser.add_argument("--compare_to", default=None, help="report agreement with this artifact")
    args = parser.parse_args(argv)

    cache_dir = Path(os.path.expanduser(args.cache_dir)) / args.env_name
    if not cache_dir.is_dir():
        parser.error(f"no cached environment at {cache_dir}")

    config = SelectionConfig(dt=args.dt, total_horizon=args.total_horizon)
    entries: Dict[str, dict] = {}
    skipped: List[str] = []

    for scene_name in _scene_names(args, cache_dir):
        try:
            metadata, frame = _load_scene(cache_dir, scene_name, dt=args.dt)
        except FileNotFoundError as error:
            skipped.append(f"{scene_name}: {error}")
            continue
        tracks = scene_tracks(
            metadata, frame, args.start_frame, args.total_horizon, args.only_types
        )
        if tracks is None:
            skipped.append(f"{scene_name}: no ego-anchored agent set at frame {args.start_frame}")
            continue
        candidate = select_adversary(tracks, ego_idx=0, config=config)
        if candidate is None:
            skipped.append(f"{scene_name}: invalid mining case, no candidate survives tiering")
            continue
        agent_names = tracks.agent_names or []
        entries[scene_name] = build_selection_entry(
            scene_name,
            candidate,
            config=config,
            agent_name=agent_names[candidate.adv_idx] if agent_names else None,
        )

    if not entries:
        print("no scene produced a selection", file=sys.stderr)
        return 1

    artifact = build_selected_vehicles(
        entries,
        metadata={
            "generator": "scripts/build_selected_vehicles.py",
            "env_name": args.env_name,
            "start_frame": args.start_frame,
            "total_horizon": args.total_horizon,
            "only_types": list(args.only_types),
        },
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")

    validated = load_selected_vehicles(output_path)
    print(f"wrote {len(validated)} selections to {output_path}")
    if skipped:
        print(f"skipped {len(skipped)} scenes; first few:")
        for line in skipped[:5]:
            print(f"  {line}")
    if args.compare_to:
        _report_comparison(validated, args.compare_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
