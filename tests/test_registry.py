import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
TBSIM_ROOT = REPOSITORY_ROOT / "third_party" / "tbsim"


class CounterSceneRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(TBSIM_ROOT))

    def test_counter_scene_has_an_independent_model_registration(self):
        from tbsim.configs.registry import get_registered_experiment_config

        config = get_registered_experiment_config("trajdata_nusc_counterscene")
        self.assertEqual(config.algo.name, "counterscene")
        self.assertEqual(config.algo.motion_dist, "cig_soft")
        self.assertTrue(config.algo.use_conflict_soft_gate)
        self.assertEqual(
            config.algo.neighbor_inds,
            [0, 1, 4, 5, 10, 11, 12, 13],
        )


if __name__ == "__main__":
    unittest.main()
