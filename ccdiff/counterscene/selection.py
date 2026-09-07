# Copyright 2026 CounterScene authors
# SPDX-License-Identifier: Apache-2.0

"""Offline conflict mining: adversary selection and guidance targets.

Implementation of CounterScene (arXiv:2603.21104) appendix A.2, "Conflict
Detection and Scoring". Conflict mining runs offline over ground-truth future
trajectories and produces the artifact consumed by
:mod:`ccdiff.counterscene.artifacts` and :mod:`ccdiff.counterscene.guidance`.

For each scene the ego is fixed at agent index 0 and every non-ego agent is
evaluated as a candidate adversary, keeping only pairs that share at least
``min_joint_steps`` jointly valid future timesteps. Per pair:

* the closest spatio-temporal encounter ``(tau_e, tau_a)`` is found by searching
  over *all* valid timestep pairs, not just simultaneous ones (eq. 10);
* the conflict point is the midpoint of that encounter (eq. 11) and ``d_min``
  its separation (eq. 12);
* the arrival-time gap (eq. 13) and the relative speed at the encounter, by
  finite differences (eq. 14), feed the type-dependent conflict score (eq. 15);
* the conflict type follows the direction cosine of the two coarse travel
  directions (eq. 16), and following conflicts split into ``rear_approach`` and
  ``lead_braking``;
* candidates are placed in tiers and the winner is chosen lexicographically by
  ``(tier, -s_conflict)``;
* the guidance weight follows eq. 17.

Scenes where no candidate survives tiering are invalid mining cases and receive
no configuration -- there is deliberately no fallback selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "ConflictCandidate",
    "SceneTracks",
    "SelectionConfig",
    "build_selected_vehicles",
    "build_selection_entry",
    "conflict_score",
    "guidance_weight",
    "iter_conflict_candidates",
    "select_adversary",
]


CONFLICT_INTERSECTION = "intersection"
CONFLICT_FOLLOWING = "following"
SUB_TYPE_REAR_APPROACH = "rear_approach"
SUB_TYPE_LEAD_BRAKING = "lead_braking"

TIER_INTERSECTION = 1
TIER_REAR_APPROACH = 2
TIER_LEAD_BRAKING = 3


@dataclass(frozen=True)
class SelectionConfig:
    """Constants of appendix A.2.

    Every value here is taken from the paper: the joint-validity requirement and
    tier thresholds from the tier-based target selection paragraph, the epsilons
    from eq. 15, the direction-cosine threshold from eq. 16, and the weights
    from eq. 17.
    """

    dt: float = 0.1
    total_horizon: int = 50
    min_joint_steps: int = 5

    # Eq. 16: cos(theta) > 0.8 is a following interaction.
    following_cos_threshold: float = 0.8

    # Eq. 15 denominators.
    intersection_eps_s: float = 0.5
    following_eps_m: float = 1.0

    # Tier-based target selection.
    min_conflict_score: float = 0.05
    tier1_max_arrival_gap_s: float = 5.0
    tier2_max_min_distance_m: float = 10.0
    tier3_max_min_distance_m: float = 12.0

    # Eq. 17.
    intersection_base_weight: float = 80.0
    intersection_score_weight: float = 40.0
    following_base_weight: float = 60.0
    following_score_weight: float = 30.0
    fallback_weight: float = -50.0

    # ------------------------------------------------------------------
    # Additional gates, all disabled by default so the defaults above stay
    # exactly appendix A.2. STRICT_CONFIG below turns them on; each one exists
    # because the published gating admits targets that cannot produce a
    # counterfactual conflict in closed loop. See docs/REPRODUCIBILITY.md.
    # ------------------------------------------------------------------

    # Eq. 15's intersection branch, v_rel / (dt + 0.5), has no distance term, and
    # tier 1 bounds only the arrival gap -- so a pair tens of metres apart scores
    # highly and is selected. Following conflicts *are* distance-bounded (tiers 2
    # and 3), which makes the omission for intersections look unintended.
    max_encounter_distance_m: Optional[float] = None

    # tier1_max_arrival_gap_s = 5.0 cannot bind on a 50-step / 5 s horizon, where
    # |tau_e - tau_a| * dt <= 4.9 s by construction. A conflict worth intervening
    # on needs the two agents at the encounter at roughly the same time.
    max_arrival_gap_s: Optional[float] = None

    # eq. 10 minimising at the last index means the true closest approach lies
    # outside the horizon: the pair is still converging when the window ends, so
    # the "encounter" is an extrapolation rather than an observed one.
    reject_boundary_encounter: bool = False

    # Symmetrically, an encounter at the first indices is unusable for a
    # different reason: the rollout starts from the observed state, so the
    # guidance term ||x_a(tau_a) - c|| evaluated on the first predicted steps has
    # almost no room to move the agent. Applied to the adversary only -- under
    # the release default adversary_only=True the ego side of both the spatial
    # and the sync term is detached, so tau_e needs no leverage.
    min_adv_encounter_index: Optional[int] = None

    # The simulator freezes agents whose GT future speed at the first step is
    # below algo_config.moving_speed_th (0.5 m/s) for the whole rollout
    # (get_stationary_mask -> preds are overwritten with zero displacement), so an
    # intervention on a parked adversary is a no-op no matter how it is guided.
    min_agent_speed_mps: Optional[float] = None

    # An agent whose GT ends mid-rollout leaves the scene, taking the conflict
    # with it; the same mask also freezes agents whose future validity is false
    # at the first step.
    require_full_horizon_validity: bool = False


DEFAULT_CONFIG = SelectionConfig()

# Mining that keeps only targets a closed-loop counterfactual can actually act
# on. Measured over the 90 published scenes, the paper's own gating yields a
# median encounter distance of 32 m, leaves 46/86 encounters pinned to the last
# horizon index, and picks a stationary adversary in 54/90 scenes -- so most
# published targets are not conflicts and the intervention cannot bite. These
# bounds are this repository's, not the paper's; keep them separate from
# DEFAULT_CONFIG so appendix A.2 stays reproducible as published.
STRICT_CONFIG = SelectionConfig(
    max_encounter_distance_m=15.0,
    max_arrival_gap_s=1.5,
    reject_boundary_encounter=True,
    min_adv_encounter_index=5,
    min_agent_speed_mps=1.0,
    require_full_horizon_validity=True,
)


@dataclass(frozen=True)
class SceneTracks:
    """Ground-truth future positions over one mining horizon, world frame.

    ``positions`` is ``(agents, steps, 2)`` and ``availability`` marks the valid
    timesteps. The agent order defines the scene-local indices written into the
    artifact, so it must match the order the simulator uses at rollout time --
    trajdata's ``SimulationScene`` takes ``scene.agent_presence[start_frame]``
    filtered by the configured agent types, preserving that order.
    """

    positions: np.ndarray
    availability: np.ndarray
    agent_names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape (agents, steps, 2)")
        availability = np.asarray(self.availability, dtype=bool)
        if availability.shape != positions.shape[:2]:
            raise ValueError("availability must have shape (agents, steps)")
        if self.agent_names is not None and len(self.agent_names) != positions.shape[0]:
            raise ValueError("agent_names must have one entry per agent")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "availability", availability)

    @property
    def num_agents(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.positions.shape[1])


@dataclass(frozen=True)
class ConflictCandidate:
    """One mined ego/candidate conflict."""

    ego_idx: int
    adv_idx: int
    conflict_type: str
    sub_type: Optional[str]
    conflict_point: Tuple[float, float]
    ego_arrival_time: int
    adv_arrival_time: int
    conflict_score: float
    tier: int
    min_distance: float
    arrival_gap: float
    relative_speed: float
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def danger_score(self) -> float:
        """Alias for the artifact's field name."""
        return self.conflict_score

    def order_key(self) -> Tuple[int, float]:
        """Lexicographic ordering over ``(tier, -s_conflict)``; smaller wins."""
        return (self.tier, -self.conflict_score)


# --------------------------------------------------------------------------
# eq. 10 - 14: the closest spatio-temporal encounter
# --------------------------------------------------------------------------


def _closest_encounter(
    ego_path: np.ndarray,
    adv_path: np.ndarray,
    ego_valid: np.ndarray,
    adv_valid: np.ndarray,
) -> Optional[Tuple[int, int, float]]:
    """Eq. 10: ``argmin`` of ``||p_e(t_e) - p_a(t_a)||`` over all valid pairs.

    The two time indices are searched independently, so the encounter need not
    be simultaneous -- that is what lets the conflict point sit off both agents'
    sampled paths.
    """
    ego_steps = np.flatnonzero(ego_valid)
    adv_steps = np.flatnonzero(adv_valid)
    if ego_steps.size == 0 or adv_steps.size == 0:
        return None
    deltas = ego_path[ego_steps][:, None, :] - adv_path[adv_steps][None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    ego_pos, adv_pos = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return (
        int(ego_steps[ego_pos]),
        int(adv_steps[adv_pos]),
        float(distances[ego_pos, adv_pos]),
    )


def _encounter_velocity(
    path: np.ndarray, valid: np.ndarray, tau: int, dt: float
) -> np.ndarray:
    """Eq. 14's per-agent term: ``(p(tau) - p(tau - 1)) / dt``.

    Falls back to a forward difference when the preceding step is unavailable.
    """
    if tau - 1 >= 0 and valid[tau - 1]:
        return (path[tau] - path[tau - 1]) / dt
    if tau + 1 < len(path) and valid[tau + 1]:
        return (path[tau + 1] - path[tau]) / dt
    return np.zeros(2, dtype=float)


def _travel_direction(path: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Coarse travel direction over the future horizon (eq. 16)."""
    steps = np.flatnonzero(valid)
    if steps.size < 2:
        return np.zeros(2, dtype=float)
    return path[steps[-1]] - path[steps[0]]


def conflict_score(
    conflict_type: str,
    *,
    relative_speed: float,
    arrival_gap: float = 0.0,
    min_distance: float = 0.0,
    config: SelectionConfig = DEFAULT_CONFIG,
) -> float:
    """Eq. 15.

    ``intersection``: ``v_rel / (dt + 0.5)`` -- emphasizes temporal co-arrival.
    ``following``:    ``v_rel / (d_min + 1.0)`` -- emphasizes closing distance.
    """
    if conflict_type == CONFLICT_INTERSECTION:
        return float(relative_speed) / (float(arrival_gap) + config.intersection_eps_s)
    if conflict_type == CONFLICT_FOLLOWING:
        return float(relative_speed) / (float(min_distance) + config.following_eps_m)
    raise ValueError(f"unknown conflict type '{conflict_type}'")


def guidance_weight(
    score: float,
    conflict_type: Optional[str],
    config: SelectionConfig = DEFAULT_CONFIG,
) -> float:
    """Eq. 17: ``-80 - 40 * min(s, 1)`` / ``-60 - 30 * min(s, 1)`` / ``-50``.

    Passing ``None`` as the type yields the paper's fallback weight. The sign is
    the artifact's convention;
    :func:`ccdiff.counterscene.guidance.build_v3_guidance` takes the absolute
    value.
    """
    if conflict_type is None:
        return float(config.fallback_weight)
    saturated = min(float(score), 1.0)
    if conflict_type == CONFLICT_INTERSECTION:
        return (
            -config.intersection_base_weight
            - config.intersection_score_weight * saturated
        )
    if conflict_type == CONFLICT_FOLLOWING:
        return -config.following_base_weight - config.following_score_weight * saturated
    raise ValueError(f"unknown conflict type '{conflict_type}'")


# --------------------------------------------------------------------------
# candidate evaluation and tiering
# --------------------------------------------------------------------------


def _assign_tier(
    conflict_type: str,
    sub_type: Optional[str],
    arrival_gap: float,
    min_distance: float,
    score: float,
    config: SelectionConfig,
) -> Optional[int]:
    """Tier-based filtering; ``None`` means the candidate is discarded."""
    if score < config.min_conflict_score:
        return None
    # Opt-in gates; no-ops under the published configuration.
    if (
        config.max_encounter_distance_m is not None
        and min_distance >= config.max_encounter_distance_m
    ):
        return None
    if config.max_arrival_gap_s is not None and arrival_gap >= config.max_arrival_gap_s:
        return None
    if conflict_type == CONFLICT_INTERSECTION:
        return TIER_INTERSECTION if arrival_gap < config.tier1_max_arrival_gap_s else None
    if sub_type == SUB_TYPE_REAR_APPROACH:
        return TIER_REAR_APPROACH if min_distance < config.tier2_max_min_distance_m else None
    if sub_type == SUB_TYPE_LEAD_BRAKING:
        return TIER_LEAD_BRAKING if min_distance < config.tier3_max_min_distance_m else None
    return None


def _evaluate_pair(
    tracks: SceneTracks,
    ego_idx: int,
    adv_idx: int,
    config: SelectionConfig,
) -> Optional[ConflictCandidate]:
    horizon = min(config.total_horizon, tracks.num_steps)
    ego_valid = tracks.availability[ego_idx, :horizon]
    adv_valid = tracks.availability[adv_idx, :horizon]
    if int(np.count_nonzero(ego_valid & adv_valid)) < config.min_joint_steps:
        return None

    # Opt-in gates that depend on the raw tracks rather than the encounter.
    if config.require_full_horizon_validity and not (
        np.all(ego_valid) and np.all(adv_valid)
    ):
        return None
    if config.min_agent_speed_mps is not None:
        for row, valid in ((ego_idx, ego_valid), (adv_idx, adv_valid)):
            steps = np.flatnonzero(valid)
            if steps.size < 2:
                return None
            first = int(steps[0])
            speed = float(
                np.linalg.norm(
                    tracks.positions[row, first + 1] - tracks.positions[row, first]
                ) / config.dt
            )
            if speed < config.min_agent_speed_mps:
                return None

    ego_path = tracks.positions[ego_idx, :horizon]
    adv_path = tracks.positions[adv_idx, :horizon]
    encounter = _closest_encounter(ego_path, adv_path, ego_valid, adv_valid)
    if encounter is None:
        return None
    tau_e, tau_a, min_distance = encounter

    if config.reject_boundary_encounter:
        last_ego = int(np.flatnonzero(ego_valid)[-1])
        last_adv = int(np.flatnonzero(adv_valid)[-1])
        if tau_e >= last_ego or tau_a >= last_adv:
            return None
    if config.min_adv_encounter_index is not None and tau_a < config.min_adv_encounter_index:
        return None

    conflict_point = 0.5 * (ego_path[tau_e] + adv_path[tau_a])  # eq. 11
    arrival_gap = abs(tau_e - tau_a) * config.dt  # eq. 13
    relative_speed = float(
        np.linalg.norm(
            _encounter_velocity(ego_path, ego_valid, tau_e, config.dt)
            - _encounter_velocity(adv_path, adv_valid, tau_a, config.dt)
        )
    )  # eq. 14

    ego_direction = _travel_direction(ego_path, ego_valid)
    adv_direction = _travel_direction(adv_path, adv_valid)
    ego_norm = float(np.linalg.norm(ego_direction))
    adv_norm = float(np.linalg.norm(adv_direction))
    if ego_norm < 1e-6 or adv_norm < 1e-6:
        cos_theta = 0.0
    else:
        cos_theta = float(ego_direction @ adv_direction / (ego_norm * adv_norm))  # eq. 16

    if cos_theta > config.following_cos_threshold:
        conflict_type = CONFLICT_FOLLOWING
        # Relative ordering of the two agents along the ego travel direction.
        first = int(np.flatnonzero(ego_valid & adv_valid)[0])
        longitudinal = float(
            (adv_path[first] - ego_path[first]) @ ego_direction / ego_norm
        )
        sub_type = SUB_TYPE_LEAD_BRAKING if longitudinal > 0.0 else SUB_TYPE_REAR_APPROACH
    else:
        conflict_type = CONFLICT_INTERSECTION
        sub_type = None

    score = conflict_score(
        conflict_type,
        relative_speed=relative_speed,
        arrival_gap=arrival_gap,
        min_distance=min_distance,
        config=config,
    )
    tier = _assign_tier(conflict_type, sub_type, arrival_gap, min_distance, score, config)
    if tier is None:
        return None

    return ConflictCandidate(
        ego_idx=ego_idx,
        adv_idx=adv_idx,
        conflict_type=conflict_type,
        sub_type=sub_type,
        conflict_point=(float(conflict_point[0]), float(conflict_point[1])),
        ego_arrival_time=tau_e,
        adv_arrival_time=tau_a,
        conflict_score=score,
        tier=tier,
        min_distance=min_distance,
        arrival_gap=arrival_gap,
        relative_speed=relative_speed,
        diagnostics={"cos_theta": cos_theta},
    )


def iter_conflict_candidates(
    tracks: SceneTracks,
    ego_idx: int = 0,
    config: SelectionConfig = DEFAULT_CONFIG,
) -> List[ConflictCandidate]:
    """Every candidate surviving tier filtering, best first."""
    if not 0 <= ego_idx < tracks.num_agents:
        raise IndexError(f"ego_idx {ego_idx} is outside the scene")
    candidates = []
    for adv_idx in range(tracks.num_agents):
        if adv_idx == ego_idx:
            continue
        candidate = _evaluate_pair(tracks, ego_idx, adv_idx, config)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=ConflictCandidate.order_key)
    return candidates


def select_adversary(
    tracks: SceneTracks,
    ego_idx: int = 0,
    config: SelectionConfig = DEFAULT_CONFIG,
) -> Optional[ConflictCandidate]:
    """The mined adversarial target, or ``None`` for an invalid mining case."""
    candidates = iter_conflict_candidates(tracks, ego_idx=ego_idx, config=config)
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------
# artifact construction
# --------------------------------------------------------------------------


def build_selection_entry(
    scene_name: str,
    candidate: ConflictCandidate,
    config: SelectionConfig = DEFAULT_CONFIG,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    """One ``selected_vehicles`` entry, in the released schema."""
    entry: Dict[str, Any] = {
        "scene_name": scene_name,
        "ego_idx": int(candidate.ego_idx),
        "adv_idx": int(candidate.adv_idx),
        "conflict_point": [
            float(candidate.conflict_point[0]),
            float(candidate.conflict_point[1]),
        ],
        "conflict_type": candidate.conflict_type,
        "sub_type": candidate.sub_type,
        "ego_arrival_time": int(candidate.ego_arrival_time),
        "adv_arrival_time": int(candidate.adv_arrival_time),
        "danger_score": float(candidate.conflict_score),
        "guidance_weight": float(
            guidance_weight(candidate.conflict_score, candidate.conflict_type, config)
        ),
        "total_horizon": int(config.total_horizon),
    }
    if agent_name is not None:
        entry["adv_agent_name"] = str(agent_name)
    return entry


def build_selected_vehicles(
    entries: Mapping[str, Mapping[str, Any]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrap per-scene entries in the ``schema_version: 1`` envelope."""
    if not entries:
        raise ValueError("refusing to build an empty selected-vehicle artifact")
    artifact: Dict[str, Any] = {
        "schema_version": 1,
        "selected_vehicles": {
            name: dict(entry) for name, entry in sorted(entries.items())
        },
    }
    if metadata:
        artifact["metadata"] = dict(metadata)
    return artifact
