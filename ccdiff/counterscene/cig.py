# Copyright 2026 CounterScene authors
# SPDX-License-Identifier: Apache-2.0

"""Conflict Interaction Graph feature construction."""

import torch


def compute_cig_features(
    relative_position,
    heading_vector,
    speed,
    extent_lw,
    availability,
    raw_ttc,
    dt=0.1,
    horizon_steps=52,
    distance_cap=20.0,
    speed_cap=20.0,
    ttc_cap=None,
):
    """Compute normalized TTI, interaction-distance and speed edge terms.

    Inputs have shape ``(batch, agents, agents, time, features)``. Delta
    position, delta velocity and TTC remain in the base CCDiff edge tensor.
    """
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if ttc_cap is None:
        ttc_cap = horizon_steps * dt

    _, num_agents, _, _, _ = relative_position.shape
    device = relative_position.device
    dtype = relative_position.dtype

    ego_speed = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1)
    ego_speed = ego_speed.unsqueeze(2).expand_as(speed)
    ego_velocity = torch.stack([ego_speed, torch.zeros_like(ego_speed)], dim=-1)
    neighbor_velocity = speed.unsqueeze(-1) * heading_vector
    delta_velocity = neighbor_velocity - ego_velocity

    offsets = torch.arange(
        1, horizon_steps + 1, device=device, dtype=dtype
    ) * dt
    offsets = offsets.view(1, 1, 1, 1, horizon_steps, 1)
    projected_relative = (
        relative_position.unsqueeze(-2) + delta_velocity.unsqueeze(-2) * offsets
    )
    projected_distance = torch.linalg.norm(projected_relative, dim=-1)

    agent_radius = 0.5 * torch.linalg.norm(extent_lw, dim=-1)
    ego_radius = torch.diagonal(agent_radius, dim1=1, dim2=2).permute(0, 2, 1)
    ego_radius = ego_radius.unsqueeze(2).expand_as(agent_radius)
    conflict_radius = agent_radius + ego_radius
    projected_clearance = projected_distance - conflict_radius.unsqueeze(-1)
    min_clearance, min_index = projected_clearance.min(dim=-1)

    tti = (min_index.to(dtype) + 1.0) * dt
    relative_speed = torch.linalg.norm(delta_velocity, dim=-1)
    ttc_score = 1.0 - torch.clamp(
        raw_ttc.squeeze(-1) / max(float(ttc_cap), 1e-6), 0.0, 1.0
    )
    tti_score = 1.0 - torch.clamp(
        tti / max(horizon_steps * dt, 1e-6), 0.0, 1.0
    )
    distance_score = 1.0 - torch.clamp(
        min_clearance.clamp(min=0.0) / max(float(distance_cap), 1e-6),
        0.0,
        1.0,
    )
    speed_score = torch.clamp(
        relative_speed / max(float(speed_cap), 1e-6), 0.0, 1.0
    )

    valid = availability.to(dtype)
    scores = [
        score * valid
        for score in (ttc_score, tti_score, distance_score, speed_score)
    ]
    diagonal = torch.arange(num_agents, device=device)
    for score in scores:
        score[:, diagonal, diagonal, :] = 0.0

    ttc_score, tti_score, distance_score, speed_score = scores
    extras = torch.stack([tti_score, distance_score, speed_score], dim=-1)
    debug_info = {
        "matrix_ttc": ttc_score[0].detach().cpu().numpy(),
        "matrix_tti": tti_score[0].detach().cpu().numpy(),
        "matrix_dint": distance_score[0].detach().cpu().numpy(),
        "matrix_vint": speed_score[0].detach().cpu().numpy(),
    }
    return extras, debug_info
