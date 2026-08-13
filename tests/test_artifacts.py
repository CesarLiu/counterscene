import json
import tempfile
import unittest
from pathlib import Path

from ccdiff.counterscene.artifacts import (
    load_selected_vehicles,
    resolve_scene_key,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
PUBLIC_ARTIFACT = REPOSITORY_ROOT / "data" / "counterscene_selected_vehicles.json"


class SelectedVehicleArtifactTest(unittest.TestCase):
    def test_published_artifact_is_complete_and_resolvable(self):
        selected = load_selected_vehicles(PUBLIC_ARTIFACT)
        self.assertEqual(len(selected), 90)
        self.assertEqual(selected["scene-0003"]["adv_idx"], 5)
        self.assertEqual(
            resolve_scene_key(3, "scene-0003", selected),
            "scene-0003",
        )

    def test_malformed_artifact_is_rejected(self):
        payload = {
            "schema_version": 1,
            "selected_vehicles": {
                "scene-0001": {"scene_name": "scene-0001"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_selected_vehicles(path)

    def test_pickle_artifacts_are_rejected(self):
        with self.assertRaises(ValueError):
            load_selected_vehicles("private-selection.pkl")


if __name__ == "__main__":
    unittest.main()
