#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Modified by the CounterScene authors, 2026.

import numpy as np
from typing import Tuple, Dict

import tbsim.utils.tensor_utils as TensorUtils

from tbsim.policies.common import Action, Plan
from tbsim.policies.base import Policy

class CCDiffHybridPolicyControl(Policy):
    '''
    Part-control wrapper for CounterScene closed-loop rollouts.

    Before intervention the controllable set is empty and the wrapper replays
    GT. After intervention it rolls out the complete scene with model outputs;
    the counterfactual gradient remains restricted upstream by
    ``set_guidance_dim(control_idx)``.
    '''
    def __init__(self, device):
        super(CCDiffHybridPolicyControl, self).__init__(device)
        self.controllable_set = [0, 1]
        self.replay_set = []

    def eval(self):
        pass

    def set_controllable_set(self, controllable_set):
        '''
        Set controllable agents index
        '''
        assert isinstance(controllable_set, list)
        self.controllable_set = controllable_set

    def set_replay_set(self, replay_set):
        '''
        Scene-local agent rows that keep their logged trajectory even after
        intervention. Used to hold a recorded ego fixed while the world model
        drives the remaining agents.
        '''
        assert isinstance(replay_set, (list, tuple))
        self.replay_set = list(replay_set)

    def add_policy(self, policy):
        self.policy = policy

    def _step_controller(self, obs, gt_pos, gt_yaw, **kwargs):
        if len(self.controllable_set) > 0:
            pred = self.policy.get_action(obs, **kwargs)[0].to_dict()
            new_pos = pred['positions']
            new_yaw = pred['yaws']
            keep = [i for i in self.replay_set if i < new_pos.shape[0]]
            if keep:
                # Hold these agents on their logged trajectory. The overwrite is
                # at the action level, so the next observation -- and hence the
                # world model's conditioning for every other agent -- already
                # reflects the logged rows.
                new_pos = new_pos.clone()
                new_yaw = new_yaw.clone()
                new_pos[keep] = gt_pos[keep].to(new_pos.dtype)
                new_yaw[keep] = gt_yaw[keep].to(new_yaw.dtype)
            gt_pos = new_pos
            gt_yaw = new_yaw

        return gt_pos, gt_yaw

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        invalid_mask = ~obs["target_availabilities"]
        gt_pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)


        gt_yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)

        edit_pos, edit_yaw = self._step_controller(obs, gt_pos, gt_yaw, **kwargs)
        edit_pos[invalid_mask] = np.nan
        edit_yaw[invalid_mask] = np.nan
        action = Action(
            positions=edit_pos,
            yaws=edit_yaw,
        )

        return action, {} # {}

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        pos = TensorUtils.to_torch(obs["target_positions"], device=self.device)
        yaw = TensorUtils.to_torch(obs["target_yaws"], device=self.device)
        edit_pos, edit_yaw = self._step_controller(obs, pos, yaw, **kwargs)

        plan = Plan(
            positions=edit_pos,
            yaws=edit_yaw,
            availabilities=TensorUtils.to_torch(obs["target_availabilities"], self.device),
        )

        return plan, {}

class ReplayPolicy(Policy):
    def __init__(self, action_log, device):
        super(ReplayPolicy, self).__init__(device)
        self.action_log = action_log

    def eval(self):
        pass

    def get_action(self, obs, step_index=None, **kwargs) -> Tuple[Action, Dict]:
        assert step_index is not None
        scene_index = TensorUtils.to_numpy(obs["scene_index"]).astype(np.int64).tolist()
        track_id = TensorUtils.to_numpy(obs["track_id"]).astype(np.int64).tolist()
        pos = []
        yaw = []
        for si, ti in zip(scene_index, track_id):
            scene_log = self.action_log[str(si)]
            if ti == -1:  # ego
                pos.append(scene_log["ego_action"]["positions"][step_index, 0])
                yaw.append(scene_log["ego_action"]["yaws"][step_index, 0])
            else:
                scene_track_id = scene_log["agents_obs"]["track_id"][0]
                agent_ind = np.where(ti == scene_track_id)[0][0]
                pos.append(scene_log["agents_action"]["positions"][step_index, agent_ind])
                yaw.append(scene_log["agents_action"]["yaws"][step_index, agent_ind])

        # stack and create the temporal dimension
        pos = np.stack(pos, axis=0)[:, None, :]
        yaw = np.stack(yaw, axis=0)[:, None, :]

        action = Action(
            positions=pos,
            yaws=yaw
        )
        return action, {}
