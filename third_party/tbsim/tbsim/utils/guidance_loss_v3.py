# Modified by the CounterScene authors, 2026.

"""CounterScene V3 conflict-point guidance.

This module lives in the vendored tbsim tree because guidance is evaluated by
``DiffuserGuidance`` during tbsim's closed-loop rollout.
"""

from __future__ import annotations

import torch
from torch import nn

from tbsim.utils.geometry_utils import transform_agents_to_world


class ConflictPointGuidanceLossV3(nn.Module):
    """Progressively move an adversary toward an ego conflict region.

    ``x`` follows the tbsim convention ``(agents, samples, steps, 6)`` with
    state channels ``(x, y, speed, yaw, acceleration, yaw rate)``.
    """

    def __init__(
        self,
        conflict_point,
        ego_arrival_time,
        adv_arrival_time,
        ego_idx,
        adv_idx,
        conflict_type="intersection",
        sub_type=None,
        danger_score=0.5,
        total_horizon=50,
        early_mult=0.2,
        mid_start_mult=0.2,
        mid_end_mult=1.5,
        late_max_mult=3.0,
        early_end=0.3,
        mid_end=0.7,
        enable_adaptive=True,
        enable_jerk=True,
        enable_conflict_aware=True,
        force_smooth_w=None,
        adversary_only=True,
    ):
        super().__init__()
        if not 0.0 <= early_end < mid_end <= 1.0:
            raise ValueError("stage boundaries must satisfy 0 <= early_end < mid_end <= 1")
        self.register_buffer(
            "conflict_point",
            torch.as_tensor(conflict_point, dtype=torch.float32),
            persistent=False,
        )
        if self.conflict_point.shape != (2,):
            raise ValueError("conflict_point must contain exactly two coordinates")

        self.ego_arrival_time = int(ego_arrival_time)
        self.adv_arrival_time = int(adv_arrival_time)
        self.orig_ego_idx = int(ego_idx)
        self.orig_adv_idx = int(adv_idx)
        self.conflict_type = str(conflict_type)
        self.sub_type = sub_type
        self.danger_score = float(danger_score)
        self.total_horizon = max(int(total_horizon), 1)
        self.early_mult = float(early_mult)
        self.mid_start_mult = float(mid_start_mult)
        self.mid_end_mult = float(mid_end_mult)
        self.late_max_mult = float(late_max_mult)
        self.early_end = float(early_end)
        self.mid_end = float(mid_end)
        self.enable_adaptive = bool(enable_adaptive)
        self.enable_jerk = bool(enable_jerk)
        self.enable_conflict_aware = bool(enable_conflict_aware)
        self.adversary_only = bool(adversary_only)

        self.spatial_w, self.temporal_w, self.smooth_w = self._get_base_weights()
        if force_smooth_w is not None:
            self.smooth_w = float(force_smooth_w)
        elif not self.enable_jerk:
            self.smooth_w = 0.0
        self.global_t = 0

    def _get_base_weights(self):
        danger = self.danger_score
        if not self.enable_conflict_aware:
            spatial, temporal, smooth = 1.5 * danger, 1.0 * danger, 0.5
        elif self.conflict_type == "intersection":
            spatial, temporal, smooth = 2.0 * danger, 1.5 * danger, 0.3
        elif self.conflict_type == "following" and self.sub_type == "rear_approach":
            spatial, temporal, smooth = 1.5 * danger, 1.0 * danger, 0.5
        elif self.conflict_type == "following" and self.sub_type == "lead_braking":
            spatial, temporal, smooth = 2.5 * danger, 0.8 * danger, 0.8
        else:
            spatial, temporal, smooth = 1.5 * danger, 1.0 * danger, 0.5
        return max(spatial, 0.3), max(temporal, 0.2), smooth

    def init_for_batch(self, example_batch):
        """Keep the interface expected by ``DiffuserGuidance``."""

    def update(self, global_t=None, **kwargs):
        if global_t is not None:
            self.global_t = int(global_t)

    def _compute_stage_multiplier(self, progress):
        if progress < self.early_end:
            return self.early_mult
        if progress < self.mid_end:
            fraction = (progress - self.early_end) / (self.mid_end - self.early_end)
            return self.mid_start_mult + fraction * (
                self.mid_end_mult - self.mid_start_mult
            )
        fraction = (progress - self.mid_end) / max(1.0 - self.mid_end, 1e-8)
        return self.mid_end_mult + min(fraction, 1.0) * (
            self.late_max_mult - self.mid_end_mult
        )

    def _compute_adaptive_arrival_times(self, ego_t, adv_t, progress, steps):
        ego_t = min(max(int(ego_t), 0), steps - 1)
        adv_t = min(max(int(adv_t), 0), steps - 1)
        if not self.enable_adaptive or progress < 0.5:
            return ego_t, adv_t

        compressed_gap = abs(ego_t - adv_t) * (1.0 - progress)
        if ego_t <= adv_t:
            adv_t = int(round(ego_t + compressed_gap))
        else:
            ego_t = int(round(adv_t + compressed_gap))
        return min(ego_t, steps - 1), min(adv_t, steps - 1)

    @staticmethod
    def _jerk_loss(trajectory):
        """Return mean third-order finite-difference magnitude per sample."""
        if trajectory.shape[-2] < 4:
            return torch.zeros(trajectory.shape[0], device=trajectory.device)
        acceleration = (
            trajectory[:, 2:] - 2.0 * trajectory[:, 1:-1] + trajectory[:, :-2]
        )
        jerk = acceleration[:, 1:] - acceleration[:, :-1]
        loss = torch.linalg.norm(jerk, dim=-1).mean(dim=-1)
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

    def forward(self, x, data_batch, agt_mask=None):
        world_from_agent = data_batch.get("world_from_agent")
        if agt_mask is not None:
            x = x[agt_mask]
            if world_from_agent is not None:
                world_from_agent = world_from_agent[agt_mask]
        if x.ndim != 4 or x.shape[-1] < 4:
            raise ValueError("x must have shape (agents, samples, steps, >=4)")

        num_agents, num_samples, num_steps, _ = x.shape
        if num_agents == 0:
            return x.new_zeros((0, num_samples))
        ego_idx = 0
        adv_idx = 0 if self.orig_ego_idx == self.orig_adv_idx else min(1, num_agents - 1)

        position = x[..., :2]
        yaw = x[..., 3:4]
        if world_from_agent is not None:
            position, _ = transform_agents_to_world(position, yaw, world_from_agent)

        progress = min(max(self.global_t / self.total_horizon, 0.0), 1.0)
        stage_multiplier = self._compute_stage_multiplier(progress)
        ego_t, adv_t = self._compute_adaptive_arrival_times(
            self.ego_arrival_time,
            self.adv_arrival_time,
            progress,
            num_steps,
        )

        ego_at_conflict = position[ego_idx, :, ego_t]
        if self.adversary_only and ego_idx != adv_idx:
            ego_at_conflict = ego_at_conflict.detach()
        adv_at_conflict = position[adv_idx, :, adv_t]
        conflict_point = self.conflict_point.to(dtype=x.dtype, device=x.device)

        if self.adversary_only:
            spatial = torch.linalg.norm(adv_at_conflict - conflict_point, dim=-1)
        else:
            spatial = torch.linalg.norm(ego_at_conflict - conflict_point, dim=-1)
            if ego_idx != adv_idx:
                spatial = spatial + torch.linalg.norm(
                    adv_at_conflict - conflict_point, dim=-1
                )
        spatial = self.spatial_w * spatial

        if ego_idx == adv_idx:
            temporal = x.new_zeros(num_samples)
        else:
            temporal = self.temporal_w * torch.linalg.norm(
                ego_at_conflict - adv_at_conflict, dim=-1
            )

        smooth = self._jerk_loss(position[adv_idx])
        if not self.adversary_only and ego_idx != adv_idx:
            smooth = smooth + self._jerk_loss(position[ego_idx])
        combined = stage_multiplier * (spatial + temporal) + self.smooth_w * smooth
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=1e6)

        loss = x.new_zeros((num_agents, num_samples))
        loss[adv_idx] = combined
        if not self.adversary_only and ego_idx != adv_idx:
            loss[ego_idx] = combined
        return loss
