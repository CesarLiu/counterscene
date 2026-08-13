import unittest

from ccdiff.counterscene.guidance import build_v3_guidance, get_v3_ablation_params


class GuidanceConfigTest(unittest.TestCase):
    def setUp(self):
        self.selection = {
            "conflict_point": [1.0, 2.0],
            "ego_arrival_time": 10,
            "adv_arrival_time": 12,
            "conflict_type": "intersection",
            "sub_type": None,
            "danger_score": 0.5,
            "guidance_weight": -100.0,
        }

    def test_v3_uses_positive_optimizer_weight(self):
        config = build_v3_guidance(self.selection, 0, 1)
        self.assertEqual(config[0]["name"], "conflict_point_guidance_v3")
        self.assertEqual(config[0]["weight"], 100.0)
        self.assertTrue(config[0]["params"]["adversary_only"])
        self.assertEqual(config[1]["agents"], "all")

    def test_minimal_ablation_disables_optional_terms(self):
        params = get_v3_ablation_params("minimal")
        self.assertFalse(params["enable_adaptive"])
        self.assertFalse(params["enable_jerk"])
        self.assertFalse(params["enable_conflict_aware"])
        self.assertEqual(params["late_max_mult"], 1.0)

    def test_unknown_ablation_fails(self):
        with self.assertRaises(ValueError):
            get_v3_ablation_params("not-a-variant")


if __name__ == "__main__":
    unittest.main()
