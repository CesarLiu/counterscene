"""Tests for the appendix A.2 conflict-mining implementation."""

import json
import math
import unittest
from pathlib import Path


try:
    import numpy as np
except ImportError:  # pragma: no cover - lightweight environment
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - lightweight environment
    torch = None


REPOSITORY_ROOT = Path(__file__).parents[1]
PUBLIC_ARTIFACT = REPOSITORY_ROOT / "data" / "counterscene_selected_vehicles.json"

DT = 0.1
STEPS = 50


def _line(start, direction, speed, steps=STEPS, accel=0.0):
    """Straight track sampled on the 10 Hz mining grid."""
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    positions = np.zeros((steps, 2))
    position = np.array(start, dtype=float)
    for index in range(steps):
        positions[index] = position
        position = position + direction * max(speed + accel * index * DT, 0.0) * DT
    return positions


def _tracks(*paths, availability=None):
    from ccdiff.counterscene.selection import SceneTracks

    positions = np.stack(paths)
    if availability is None:
        availability = np.ones(positions.shape[:2], dtype=bool)
    return SceneTracks(positions=positions, availability=availability)


@unittest.skipIf(np is None, "numpy is not installed in the lightweight test environment")
class GuidanceWeightTest(unittest.TestCase):
    """Eq. 17."""

    def test_reproduces_every_published_weight(self):
        from ccdiff.counterscene.selection import guidance_weight

        with PUBLIC_ARTIFACT.open("r", encoding="utf-8") as handle:
            published = json.load(handle)["selected_vehicles"]
        self.assertEqual(len(published), 90)
        for name, entry in published.items():
            with self.subTest(scene=name):
                self.assertAlmostEqual(
                    guidance_weight(entry["danger_score"], entry["conflict_type"]),
                    entry["guidance_weight"],
                    places=9,
                )

    def test_saturates_at_score_one(self):
        from ccdiff.counterscene.selection import guidance_weight

        self.assertEqual(guidance_weight(0.0, "intersection"), -80.0)
        self.assertEqual(guidance_weight(0.5, "intersection"), -100.0)
        self.assertEqual(guidance_weight(1.0, "intersection"), -120.0)
        self.assertEqual(guidance_weight(40.0, "intersection"), -120.0)
        self.assertEqual(guidance_weight(0.0, "following"), -60.0)
        self.assertEqual(guidance_weight(1.0, "following"), -90.0)
        self.assertEqual(guidance_weight(40.0, "following"), -90.0)

    def test_fallback_weight(self):
        from ccdiff.counterscene.selection import guidance_weight

        self.assertEqual(guidance_weight(3.0, None), -50.0)

    def test_unknown_type_is_rejected(self):
        from ccdiff.counterscene.selection import guidance_weight

        with self.assertRaises(ValueError):
            guidance_weight(1.0, "merging")


@unittest.skipIf(np is None, "numpy is not installed in the lightweight test environment")
class ConflictScoreTest(unittest.TestCase):
    """Eq. 15."""

    def test_intersection_branch(self):
        from ccdiff.counterscene.selection import conflict_score

        self.assertAlmostEqual(
            conflict_score("intersection", relative_speed=6.0, arrival_gap=1.0),
            6.0 / 1.5,
        )

    def test_following_branch(self):
        from ccdiff.counterscene.selection import conflict_score

        self.assertAlmostEqual(
            conflict_score("following", relative_speed=4.0, min_distance=3.0),
            1.0,
        )


@unittest.skipIf(np is None, "numpy is not installed in the lightweight test environment")
class EncounterGeometryTest(unittest.TestCase):
    """Eq. 10 - 14."""

    def test_encounter_search_is_not_simultaneous(self):
        from ccdiff.counterscene.selection import select_adversary

        # The ego trails the adversary's lane position the whole time, so the
        # closest *simultaneous* distance is large, while the closest cross-time
        # pair is the 2 m lateral offset at (ego t=40, adv t=0).
        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        adv = _line((40.0, 2.0), (1.0, 0.0), 6.0)
        simultaneous = float(np.linalg.norm(ego - adv, axis=1).min())
        candidate = select_adversary(_tracks(ego, adv))

        self.assertIsNotNone(candidate)
        self.assertGreater(simultaneous, 15.0)
        self.assertAlmostEqual(candidate.min_distance, 2.0, places=6)
        self.assertEqual(candidate.ego_arrival_time, 40)
        self.assertEqual(candidate.adv_arrival_time, 0)

    def test_conflict_point_is_the_encounter_midpoint(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        adv = _line((40.0, 2.0), (1.0, 0.0), 6.0)
        candidate = select_adversary(_tracks(ego, adv))

        expected = 0.5 * (
            ego[candidate.ego_arrival_time] + adv[candidate.adv_arrival_time]
        )
        self.assertAlmostEqual(candidate.conflict_point[0], expected[0], places=6)
        self.assertAlmostEqual(candidate.conflict_point[1], expected[1], places=6)

    def test_arrival_gap_and_relative_speed(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        adv = _line((40.0, 2.0), (1.0, 0.0), 6.0)
        candidate = select_adversary(_tracks(ego, adv))

        gap = abs(candidate.ego_arrival_time - candidate.adv_arrival_time) * DT
        self.assertAlmostEqual(candidate.arrival_gap, gap, places=9)
        # Both travel along +x, so v_rel is the speed difference.
        self.assertAlmostEqual(candidate.relative_speed, 4.0, places=5)

    def test_pairs_without_five_joint_steps_are_skipped(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((-20.0, 0.0), (1.0, 0.0), 8.0)
        crossing = _line((0.0, -20.0), (0.0, 1.0), 8.0)
        availability = np.ones((2, STEPS), dtype=bool)
        availability[1, 4:] = False  # only four jointly valid steps
        self.assertIsNone(
            select_adversary(_tracks(ego, crossing, availability=availability))
        )


@unittest.skipIf(np is None, "numpy is not installed in the lightweight test environment")
class ConflictTypeTest(unittest.TestCase):
    """Eq. 16 and the following sub-types."""

    def test_crossing_directions_are_an_intersection(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((-20.0, 0.0), (1.0, 0.0), 8.0)
        crossing = _line((0.0, -20.0), (0.0, 1.0), 8.0)
        candidate = select_adversary(_tracks(ego, crossing))

        self.assertEqual(candidate.conflict_type, "intersection")
        self.assertIsNone(candidate.sub_type)
        self.assertEqual(candidate.tier, 1)

    def test_direction_cosine_threshold_is_08(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        # cos(theta) = 0.9 > 0.8 -> following; 0.7 < 0.8 -> intersection.
        for cosine, expected in ((0.9, "following"), (0.7, "intersection")):
            with self.subTest(cosine=cosine):
                direction = (cosine, math.sqrt(1.0 - cosine**2))
                adv = _line((6.0, 0.0), direction, 6.0)
                candidate = select_adversary(_tracks(ego, adv))
                self.assertIsNotNone(candidate)
                self.assertEqual(candidate.conflict_type, expected)

    def test_trailing_adversary_is_a_rear_approach(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        adv = _line((-5.0, 4.0), (1.0, 0.0), 6.0)
        candidate = select_adversary(_tracks(ego, adv))

        self.assertEqual(candidate.conflict_type, "following")
        self.assertEqual(candidate.sub_type, "rear_approach")
        self.assertEqual(candidate.tier, 2)

    def test_leading_adversary_is_lead_braking(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        adv = _line((5.0, 4.0), (1.0, 0.0), 6.0)
        candidate = select_adversary(_tracks(ego, adv))

        self.assertEqual(candidate.conflict_type, "following")
        self.assertEqual(candidate.sub_type, "lead_braking")
        self.assertEqual(candidate.tier, 3)


@unittest.skipIf(np is None, "numpy is not installed in the lightweight test environment")
class TieringTest(unittest.TestCase):
    """Tier-based target selection."""

    def test_rear_approach_and_lead_braking_have_different_distance_bounds(self):
        from ccdiff.counterscene.selection import select_adversary

        # Parallel lane 11 m away: inside tier 3's 12 m bound, outside tier 2's 10 m.
        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        trailing = _line((-5.0, 11.0), (1.0, 0.0), 6.0)
        leading = _line((5.0, 11.0), (1.0, 0.0), 6.0)

        self.assertIsNone(select_adversary(_tracks(ego, trailing)))
        accepted = select_adversary(_tracks(ego, leading))
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.sub_type, "lead_braking")
        self.assertAlmostEqual(accepted.min_distance, 11.0, places=6)

    def test_low_scoring_candidates_are_discarded(self):
        from ccdiff.counterscene.selection import SelectionConfig, select_adversary

        ego = _line((-20.0, 0.0), (1.0, 0.0), 8.0)
        crossing = _line((0.0, -20.0), (0.0, 1.0), 8.0)
        tracks = _tracks(ego, crossing)

        self.assertIsNotNone(select_adversary(tracks))
        strict = SelectionConfig(min_conflict_score=1e6)
        self.assertIsNone(select_adversary(tracks, config=strict))

    def test_default_score_floor_is_005(self):
        from ccdiff.counterscene.selection import SelectionConfig

        self.assertEqual(SelectionConfig().min_conflict_score, 0.05)

    def test_lower_tier_wins_over_a_higher_score(self):
        from ccdiff.counterscene.selection import iter_conflict_candidates

        ego = _line((0.0, 0.0), (1.0, 0.0), 1.0)
        # Slow crossing agent meeting the ego late: intersection (tier 1), but a
        # small score because v_rel is low and the arrival gap is nearly 5 s.
        crossing = _line((4.9, -0.2), (0.0, 1.0), 1.0)
        # Adjacent-lane leader pulling away: following (tier 3), much higher score.
        leading = _line((2.0, 1.0), (1.0, 0.0), 5.0)

        candidates = iter_conflict_candidates(_tracks(ego, crossing, leading))
        tiers = {candidate.tier for candidate in candidates}
        self.assertIn(1, tiers)
        self.assertIn(3, tiers)

        winner = candidates[0]
        self.assertEqual(winner.tier, 1)
        loser = next(c for c in candidates if c.tier == 3)
        self.assertGreater(loser.conflict_score, winner.conflict_score)

    def test_scene_without_survivors_is_an_invalid_mining_case(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((0.0, 0.0), (1.0, 0.0), 10.0)
        far = _line((0.0, 400.0), (1.0, 0.0), 10.0)  # identical motion, v_rel = 0
        self.assertIsNone(select_adversary(_tracks(ego, far)))

    def test_ego_index_outside_the_scene_is_rejected(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((0.0, 0.0), (1.0, 0.0), 8.0)
        other = _line((0.0, -20.0), (0.0, 1.0), 8.0)
        with self.assertRaises(IndexError):
            select_adversary(_tracks(ego, other), ego_idx=7)


@unittest.skipIf(np is None, "numpy is not installed in the lightweight test environment")
class ArtifactTest(unittest.TestCase):
    def _candidate(self):
        from ccdiff.counterscene.selection import select_adversary

        ego = _line((-20.0, 0.0), (1.0, 0.0), 8.0)
        crossing = _line((0.0, -20.0), (0.0, 1.0), 8.0)
        return select_adversary(_tracks(ego, crossing))

    def test_generated_entries_pass_the_released_validator(self):
        import tempfile

        from ccdiff.counterscene.artifacts import load_selected_vehicles
        from ccdiff.counterscene.selection import (
            build_selected_vehicles,
            build_selection_entry,
        )

        artifact = build_selected_vehicles(
            {"scene-0001": build_selection_entry("scene-0001", self._candidate())}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_vehicles.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(artifact, handle)
            loaded = load_selected_vehicles(path)

        self.assertEqual(sorted(loaded), ["scene-0001"])
        self.assertEqual(loaded["scene-0001"]["ego_idx"], 0)
        self.assertEqual(loaded["scene-0001"]["adv_idx"], 1)

    def test_generated_entry_drives_the_v3_guidance_builder(self):
        from ccdiff.counterscene.guidance import build_v3_guidance
        from ccdiff.counterscene.selection import build_selection_entry

        entry = build_selection_entry("scene-0001", self._candidate())
        guidance = build_v3_guidance(entry, 0, 1)
        self.assertEqual(guidance[0]["name"], "conflict_point_guidance_v3")
        self.assertGreater(guidance[0]["weight"], 0.0)
        self.assertEqual(guidance[0]["params"]["conflict_type"], "intersection")

    @unittest.skipIf(torch is None, "torch is not installed in the lightweight test environment")
    def test_generated_targets_construct_the_real_v3_loss(self):
        import sys

        sys.path.insert(0, str(REPOSITORY_ROOT / "third_party" / "tbsim"))
        from tbsim.utils.guidance_loss_v3 import ConflictPointGuidanceLossV3

        from ccdiff.counterscene.guidance import build_v3_guidance
        from ccdiff.counterscene.selection import build_selection_entry

        entry = build_selection_entry("scene-0001", self._candidate())
        guidance = build_v3_guidance(entry, 0, 1)[0]

        loss_fn = ConflictPointGuidanceLossV3(**guidance["params"])
        self.assertEqual(loss_fn.conflict_type, "intersection")
        self.assertEqual(loss_fn.total_horizon, entry["total_horizon"])

        # (agents, samples, steps, 6): x, y, speed, yaw, accel, yaw rate.
        trajectory = torch.zeros(2, 1, STEPS, 6, requires_grad=True)
        loss_fn.update(global_t=25)
        loss = loss_fn(trajectory, {})
        loss.sum().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(float(trajectory.grad[1].abs().sum()), 0.0)
        self.assertEqual(float(trajectory.grad[0].abs().sum()), 0.0)

    def test_empty_artifacts_are_refused(self):
        from ccdiff.counterscene.selection import build_selected_vehicles

        with self.assertRaises(ValueError):
            build_selected_vehicles({})


if __name__ == "__main__":
    unittest.main()
