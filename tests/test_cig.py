import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the lightweight test environment")
class ConflictInteractionGraphTest(unittest.TestCase):
    def test_features_have_expected_shape_and_zero_diagonal(self):
        from ccdiff.counterscene.cig import compute_cig_features

        batch, agents, steps = 1, 2, 3
        relative_position = torch.zeros(batch, agents, agents, steps, 2)
        relative_position[:, 0, 1, :, 0] = 5.0
        relative_position[:, 1, 0, :, 0] = -5.0
        heading_vector = torch.zeros_like(relative_position)
        heading_vector[..., 0] = 1.0
        speed = torch.ones(batch, agents, agents, steps)
        extent_lw = torch.ones_like(relative_position)
        availability = torch.ones(batch, agents, agents, steps, dtype=torch.bool)
        raw_ttc = torch.full((batch, agents, agents, steps, 1), 5.0)

        features, debug = compute_cig_features(
            relative_position,
            heading_vector,
            speed,
            extent_lw,
            availability,
            raw_ttc,
            horizon_steps=5,
        )

        self.assertEqual(features.shape, (batch, agents, agents, steps, 3))
        self.assertTrue(torch.all(features[:, 0, 0] == 0))
        self.assertTrue(torch.all(features[:, 1, 1] == 0))
        self.assertEqual(
            set(debug),
            {"matrix_ttc", "matrix_tti", "matrix_dint", "matrix_vint"},
        )


if __name__ == "__main__":
    unittest.main()
