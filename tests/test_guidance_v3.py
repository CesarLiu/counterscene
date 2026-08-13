import sys
import unittest
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the lightweight test environment")
class GuidanceLossV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).parents[1] / "third_party" / "tbsim"))
        from tbsim.utils.guidance_loss_v3 import ConflictPointGuidanceLossV3

        cls.loss_type = ConflictPointGuidanceLossV3

    def test_adversary_only_has_no_ego_gradient(self):
        loss_fn = self.loss_type(
            conflict_point=[1.0, 0.0],
            ego_arrival_time=2,
            adv_arrival_time=2,
            ego_idx=0,
            adv_idx=1,
            enable_jerk=False,
            adversary_only=True,
        )
        trajectories = torch.zeros(2, 1, 5, 6, requires_grad=True)
        loss_fn(trajectories, {}).sum().backward()
        self.assertEqual(float(trajectories.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(trajectories.grad[1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
