#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Modified by the CounterScene authors, 2026.

"""A script for evaluating closed-loop simulation"""
import argparse
import numpy as np
import json
import random
import yaml
import importlib
from pprint import pprint

import os
import torch

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
from tbsim.evaluation.env_builders import EnvUnifiedBuilder
from tbsim.policies.wrappers import (
    RolloutWrapper,
    Pos2YawWrapper,
)

from tbsim.utils.tensor_utils import map_ndarray
import tbsim.utils.tensor_utils as TensorUtils
from ccdiff.counterscene.artifacts import (
    load_selected_vehicles,
    resolve_scene_key,
)
from ccdiff.counterscene.guidance import build_v3_guidance


def run_scene_editor(eval_cfg, save_cfg, data_to_disk, render_to_video, render_to_img, render_cfg, part_control=False, controllable_agent=-1,
                    num_steps_intervention=0, selected_vehicles=None,
                    conflict_ablation="full",
                    conflict_adversary_only=True):
    assert eval_cfg.env in ["nusc", "trajdata"], "Currently only nusc and trajdata environments are supported"

    set_global_batch_type("trajdata")
    if eval_cfg.env == "nusc":
        set_global_trajdata_batch_env("nusc_trainval")
    elif eval_cfg.env == "trajdata":
        # assumes all used trajdata datasets use share same map layers
        set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])

    # print(eval_cfg)

    # for reproducibility
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)
        pass


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    # create policy and rollout wrapper
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    if part_control:
        composer_class = getattr(policy_composers, 'CCDiffHierarchicalPolicy')
        composer = composer_class(eval_cfg, device)
        control_policy, exp_config = composer.get_policy()
        # control_policy.add_policy(policy)


    # determines cfg for rasterizing agents
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)

    # print(exp_config.algo)
    # ----------------------------------------------------------------------------------
    policy_model = None
    print('policy', policy)
    if hasattr(policy, 'model'):
        policy_model = policy.model

    # Set evaluation time sampling/optimization parameters
    if eval_cfg.apply_guidance:
        if eval_cfg.eval_class in ['CCDiff', 'Diffuser', 'TrafficSim', 'BC', 'HierarchicalSampleNew']:
            # policy_model.nets["policy"].controllable_agent = controllable_agent
            # policy_model.nets["policy"].current_perturbation_guidance.controlled_agent = controllable_agent
            # print('set agent number to: ', controllable_agent)
            # policy_model.nets["policy"].reset_guidance()
            policy_model.set_guidance_optimization_params(eval_cfg.guidance_optimization_params)

        if eval_cfg.eval_class in ['CCDiff', 'Diffuser']:
            # policy_model.nets["policy"].controllable_agent = controllable_agent
            # policy_model.nets["policy"].current_perturbation_guidance.controlled_agent = controllable_agent
            # print('set agent number to: ', controllable_agent)
            # policy_model.nets["policy"].reset_guidance()
            policy_model.set_diffusion_specific_params(eval_cfg.diffusion_specific_params)
    # ----------------------------------------------------------------------------------

    # create env
    if eval_cfg.env == "trajdata":
        env_builder = EnvUnifiedBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        env = env_builder.get_env()
    else:
        raise NotImplementedError("{} is not a valid env".format(eval_cfg.env))

    # eval loop
    obs_to_torch = eval_cfg.eval_class not in [
        "GroundTruth",
        "GroundTruthOpenLoop",
        "GroundTruthNaN",
        "CCDiffHierarchicalPolicy",
        "ReplayAction",
    ]

    heuristic_config = None
    use_ui = False
    if "ui" in eval_cfg.edits.editing_source:
        # TODO if using UI, initialize UI
        print("Using ONLY user interface to get scene edits...")
        use_ui = True
        raise NotImplementedError('UI')
    elif "heuristic" in eval_cfg.edits.editing_source:
        # verify heuristic args are valid
        if eval_cfg.edits.heuristic_config is not None:
            heuristic_config = eval_cfg.edits.heuristic_config
        else:
            heuristic_config = []

    render_rasterizer = None
    if render_to_video or render_to_img:
        from tbsim.utils.scene_edit_utils import get_trajdata_renderer
        # initialize rasterizer once for all scenes
        render_rasterizer = get_trajdata_renderer(eval_cfg.trajdata_source_test,
                                                  eval_cfg.trajdata_data_dirs,
                                                  future_sec=eval_cfg.future_sec,
                                                  history_sec=eval_cfg.history_sec,
                                                  raster_size=render_cfg['size'],
                                                  px_per_m=render_cfg['px_per_m'],
                                                  rebuild_maps=False,
                                                  cache_location='~/.unified_data_cache')

    result_stats = None
    scene_i = 0
    eval_scenes = eval_cfg.eval_scenes
    if part_control and selected_vehicles is None:
        raise ValueError("--part_control requires --selected_vehicles_path")
    if selected_vehicles is not None and not part_control:
        raise ValueError("CounterScene selected-vehicle guidance requires --part_control")
    if selected_vehicles is not None and eval_cfg.num_scenes_per_batch != 1:
        raise ValueError("CounterScene evaluation requires --num_scenes_per_batch 1")
    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch
        # check to make sure all the scenes are valid at starting step
        scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=None)
        if part_control:
            requested_agents = max(controllable_agent, 2)
            max_controllable_size = min(requested_agents, env.current_num_agents)
            current_names = getattr(env, "current_scene_names", [])
            current_name = current_names[0] if current_names else None
            selected_key = resolve_scene_key(
                scene_indices[0], current_name, selected_vehicles
            )
            if selected_key is None:
                print(
                    f"No published vehicle selection for "
                    f"{current_name or scene_indices[0]}, skipping"
                )
                torch.cuda.empty_cache()
                continue
            selection = selected_vehicles[selected_key]
            control_idx = [
                int(selection["ego_idx"]),
                int(selection["adv_idx"]),
            ]
            control_idx = list(dict.fromkeys(control_idx))[:max_controllable_size]
            if any(
                index < 0 or index >= env.current_num_agents
                for index in control_idx
            ):
                raise IndexError(
                    f"scene {selected_key}: selected vehicle index is outside the environment"
                )

            if len(control_idx) < max_controllable_size:
                additional_idx = list(np.random.choice([i for i in range(env.current_num_agents) if i not in control_idx],
                                size=(max_controllable_size-len(control_idx))))
                control_idx = control_idx + additional_idx

            mark_agents = [(i, "*", "red") for i in control_idx]
            control_policy.set_controllable_set(control_idx)
            if eval_cfg.eval_class in ['CCDiff']:
                policy_model.nets['policy'].set_guidance_dim(control_idx, eval_cfg.n_step_action)
                print("set guidance dim: ", control_idx, 'timestep: ', eval_cfg.n_step_action)
            else:
                policy_model.nets['policy'].set_guidance_dim(control_idx)
                print("set guidance dim: ", control_idx,)

        else:
            mark_agents = []
        scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
        if len(scene_indices) == 0:
            print('no valid scenes in this batch, skipping...')
            torch.cuda.empty_cache()
            continue


        # if requested, split each scene up into multiple simulations
        start_frame_index = [[exp_config.algo.history_num_frames+1]] * len(scene_indices)
        if eval_cfg.num_sim_per_scene > 1:
            start_frame_index = []
            for si in range(len(scene_indices)):
                cur_scene = env._current_scenes[si].scene
                sframe = exp_config.algo.history_num_frames+1
                # want to make sure there's GT for the full rollout
                eframe = cur_scene.length_timesteps - eval_cfg.num_simulation_steps
                scene_frame_inds = np.linspace(sframe, eframe, num=eval_cfg.num_sim_per_scene, dtype=int).tolist()
                start_frame_index.append(scene_frame_inds)

        # how many sims to run for the current batch of scenes
        print('Starting frames in current scenes:', start_frame_index)
        for ei in range(eval_cfg.num_sim_per_scene):
            guidance_config = None   # for the current batch of scenes
            constraint_config = None # for the current batch of scenes

            cur_start_frames = [scene_start[ei] for scene_start in start_frame_index]
            # double check all scenes are valid at the current start step
            scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=cur_start_frames)
            sim_scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
            sim_start_frames = [sframe for sframe, sval in zip(cur_start_frames, scenes_valid) if sval]
            if len(sim_scene_indices) == 0:
                torch.cuda.empty_cache()
                continue

            if not use_ui:
                # getting edits from either the config file or on-the-fly heuristics
                if "config" in eval_cfg.edits.editing_source:
                    guidance_config = eval_cfg.edits.guidance_config
                    constraint_config  = eval_cfg.edits.constraint_config
                if "heuristic" in eval_cfg.edits.editing_source:
                    # reset so that we can get an example batch to initialize guidance more efficiently
                    env.reset(scene_indices=scene_indices, start_frame_index=sim_start_frames)
                    ex_obs = env.get_observation()
                    if obs_to_torch:
                        device = policy.device if device is None else device
                        ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)

                    # build heuristic guidance configs for these scenes
                    heuristic_guidance_cfg = compute_heuristic_guidance(heuristic_config,
                                                                        env,
                                                                        sim_scene_indices,
                                                                        sim_start_frames,
                                                                        example_batch=ex_obs['agents'])

                    if len(heuristic_config) > 0:
                        # we asked to apply some guidance, but if heuristic determined there was no valid
                        #       guidance to apply (e.g. no social groups), we should skip these scenes.
                        valid_scene_inds = []
                        for sci, sc_cfg in enumerate(heuristic_guidance_cfg):
                            if len(sc_cfg) > 0:
                                valid_scene_inds.append(sci)

                        # collect only valid scenes under the given heuristic config
                        heuristic_guidance_cfg = [heuristic_guidance_cfg[vi] for vi in valid_scene_inds]
                        sim_scene_indices = [sim_scene_indices[vi] for vi in valid_scene_inds]
                        sim_start_frames = [sim_start_frames[vi] for vi in valid_scene_inds]
                        # skip if no valid...
                        if len(sim_scene_indices) == 0:
                            print('No scenes with valid heuristic configs in this sim, skipping...')
                            torch.cuda.empty_cache()
                            continue

                    # add to the current guidance config
                    guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)
                if selected_vehicles is not None:
                    conflict_guidance = []
                    sim_names = getattr(env, "current_scene_names", [])
                    for scene_position, scene_index in enumerate(sim_scene_indices):
                        current_name = (
                            sim_names[scene_position]
                            if scene_position < len(sim_names)
                            else None
                        )
                        selected_key = resolve_scene_key(
                            scene_index, current_name, selected_vehicles
                        )
                        if selected_key is None:
                            conflict_guidance.append([])
                            continue
                        selection = selected_vehicles[selected_key]
                        ego_global = int(selection["ego_idx"])
                        adv_global = int(selection["adv_idx"])
                        if ego_global not in control_idx or adv_global not in control_idx:
                            raise ValueError(
                                f"scene {selected_key}: selected pair is not controllable"
                            )
                        conflict_guidance.append(
                            build_v3_guidance(
                                selection,
                                control_idx.index(ego_global),
                                control_idx.index(adv_global),
                                ablation=conflict_ablation,
                                total_horizon=eval_cfg.num_simulation_steps,
                                adversary_only=conflict_adversary_only,
                            )
                        )
                    guidance_config = merge_guidance_configs(
                        guidance_config, conflict_guidance
                    )
            else:
                # TODO get guidance from the UI
                # TODO for UI, get edits from user. loop continuously until the user presses
                #       "play" or something like that then we roll out.
                raise NotImplementedError()
        if len(sim_scene_indices) == 0:
            print('No scenes with valid heuristic configs in this scene, skipping...')
            torch.cuda.empty_cache()
            continue

        # remove agents from agent_collision guidance if they are in chosen gptcollision pair
        for si in range(len(guidance_config)):
            if len(guidance_config[si]) > 0:
                agent_collision_heur_ind = None
                gpt_collision_heur_ind = None
                for i, cur_heur in enumerate(guidance_config[si]):
                    if cur_heur['name'] == 'agent_collision':
                        agent_collision_heur_ind = i
                    elif cur_heur['name'] == 'gptcollision':
                        gpt_collision_heur_ind = i
                if agent_collision_heur_ind is not None and gpt_collision_heur_ind is not None:
                        ind1 = guidance_config[si][gpt_collision_heur_ind]['params']['target_ind']
                        ind2 = guidance_config[si][gpt_collision_heur_ind]['params']['ref_ind']
                        excluded_agents = [ind1, ind2]
                        guidance_config[si][agent_collision_heur_ind]['params']['excluded_agents'] = excluded_agents
                        print('excluded_agents', excluded_agents)

        # ----------------------------------------------------------------------------------
        # Sampling Wrapper leveraging most existing policy composer sampling interfaces
        from tbsim.policies.wrappers import NewSamplingPolicyWrapper
        if eval_cfg.eval_class in ['TrafficSim', 'HierarchicalSampleNew']:
            if scene_i == eval_cfg.num_scenes_per_batch or not isinstance(policy, NewSamplingPolicyWrapper):
                policy = NewSamplingPolicyWrapper(policy, guidance_config)
            else:
                policy.update_guidance_config(guidance_config)
        # ----------------------------------------------------------------------------------

        if eval_cfg.policy.pos_to_yaw:
            policy = Pos2YawWrapper(
                policy,
                dt=exp_config.algo.step_time,
                yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
            )
        # right now assume control of full scene
        if part_control:
            control_policy.add_policy(policy)
            rollout_policy = RolloutWrapper(agents_policy=control_policy)
            # policy = RolloutWrapper(agents_policy=policy)
        else:
            rollout_policy = RolloutWrapper(agents_policy=policy)
        stats, info, renderings = guided_rollout(
            env,
            rollout_policy,
            policy_model,
            n_step_action=eval_cfg.n_step_action,
            guidance_config=guidance_config,
            constraint_config=constraint_config,
            render=False, # render after the fact
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch,
            horizon=eval_cfg.num_simulation_steps,
            start_frames=sim_start_frames,
            eval_class=eval_cfg.eval_class,
            apply_guidance=eval_cfg.apply_guidance,
            num_steps_intervention=num_steps_intervention
        )

        print(info["scene_index"])
        print(sim_start_frames)
        pprint(stats)

        # aggregate stats from the same class of guidance within each scene
        #       this helps parse_scene_edit_results
        guide_agg_dict = {}
        pop_list = []
        for k,v in stats.items():
            if k.split('_')[0] == 'guide':
                guide_name = '_'.join(k.split('_')[:-1])
                guide_scene_tag = k.split('_')[-1][:2]
                canon_name = guide_name + '_%sg0' % (guide_scene_tag)
                if canon_name not in guide_agg_dict:
                    guide_agg_dict[canon_name] = []
                guide_agg_dict[canon_name].append(v)
                # remove from stats
                pop_list.append(k)
        for k in pop_list:
            stats.pop(k, None)
        # average over all of the same guide stats in each scene
        for k,v in guide_agg_dict.items():
            scene_stats = np.stack(v, axis=0) # guide_per_scenes x num_scenes (all are nan except 1)
            stats[k] = np.mean(scene_stats, axis=0)

        # aggregate metrics stats
        if result_stats is None:
            result_stats = stats
            result_stats["scene_index"] = np.array(info["scene_index"])
        else:
            for k in stats:
                if k not in result_stats:
                    result_stats[k] = stats[k]
                else:
                    result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
            result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

        # write stats to disk
        with open(os.path.join(eval_cfg.results_dir, "stats.json"), "w+") as fp:
            stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
            json.dump(stats_to_write, fp)
        if "matrix_ttc" in info.keys() and "matrix_dist" in info.keys():
            os.makedirs(os.path.join(eval_cfg.results_dir, "matrix"), exist_ok=True)
            for matrix_name in ("ttc", "tti", "dint", "vint", "dist"):
                info_key = "matrix_" + matrix_name
                if info_key in info:
                    np.save(
                        os.path.join(
                            eval_cfg.results_dir,
                            "matrix",
                            "{}_{}.npy".format(matrix_name, info["scene_index"]),
                        ),
                        info[info_key],
                    )

        if render_to_video or render_to_img:
            # high quality
            from tbsim.utils.scene_edit_utils import visualize_guided_rollout
            scene_cnt = 0
            for si, scene_buffer in zip(info["scene_index"], info["buffer"]):
                viz_dir = os.path.join(eval_cfg.results_dir, "viz/")
                invalid_guidance = guidance_config is None or len(guidance_config) == 0
                invalid_constraint = constraint_config is None or len(constraint_config) == 0
                visualize_guided_rollout(viz_dir, render_rasterizer, si, scene_buffer,
                                            guidance_config=None if invalid_guidance else guidance_config[scene_cnt],
                                            constraint_config=None if invalid_constraint else constraint_config[scene_cnt],
                                            fps=(1.0 / exp_config.algo.step_time),
                                            n_step_action=eval_cfg.n_step_action,
                                            viz_diffusion_steps=False,
                                            first_frame_only=render_to_img,
                                            sim_num=sim_start_frames[scene_cnt],
                                            save_every_n_frames=render_cfg['save_every_n_frames'],
                                            draw_mode=render_cfg['draw_mode'],
                                            mark_agents=mark_agents)
                scene_cnt += 1

        if data_to_disk and "buffer" in info:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                sim_start_frames,
                h5_path=eval_cfg.experience_hdf5_path
            )
        torch.cuda.empty_cache()


def dump_episode_buffer(buffer, scene_index, start_frames, h5_path):
    import h5py
    h5_file = h5py.File(h5_path, "a")

    for ei, si, scene_buffer in zip(start_frames, scene_index, buffer):
        for mk in scene_buffer:
            h5key = "/{}_{}/{}".format(si, ei, mk)
            h5_file.create_dataset(h5key, data=scene_buffer[mk])
    h5_file.close()
    print("scene {} written to {}".format(scene_index, h5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "trajdata"],
        help="Which env to run editing in",
        required=True
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )

    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ------ for BITS ------
    parser.add_argument(
        "--planner_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--planner_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )

    parser.add_argument(
        "--predictor_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--predictor_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ----------------------

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default=None,
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root directory of the dataset"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=None,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="whether to render videos"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--registered_name",
        type=str,
        default='trajdata_nusc_counterscene',
    )

    parser.add_argument(
        "--render_img",
        action="store_true",
        default=False,
        help="whether to only render the first frame of rollout"
    )

    parser.add_argument(
        "--render_size",
        type=int,
        default=400,
        help="width and height of the rendered image size in pixels"
    )

    parser.add_argument(
        "--render_px_per_m",
        type=float,
        default=2.0,
        help="resolution of rendering"
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=5,
        help="saving videos while skipping every n frames"
    )

    parser.add_argument(
        "--draw_mode",
        type=str,
        default='action',
        help="['action', 'entire_traj', 'map']"
    )

    #
    # Editing options
    #
    parser.add_argument(
        "--editing_source",
        type=str,
        choices=["config", "heuristic", "conflict", "ui", "none"],
        nargs="+",
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics. UI will use interactive interface. \
              config and heuristic may be used together. If none, does not use edits."
    )
    parser.add_argument(
        "--controllable_agent",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--part_control",
        action="store_true",
    )
    parser.add_argument(
        "--n_step_action",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--w_collision",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--w_offroad",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num_steps_intervention",
        type=int,
        default=0,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )
    parser.add_argument(
        "--num_simulation_steps",
        type=int,
        default=None,
        help="Closed-loop horizon in 10 Hz frames (50 corresponds to 5 seconds)",
    )

    parser.add_argument(
        "--selected_vehicles_path",
        type=str,
        default="data/counterscene_selected_vehicles.json",
        help="Published CounterScene vehicle pairs and guidance targets (JSON)",
    )
    parser.add_argument(
        "--v3_ablation_variant",
        type=str,
        default="full",
        choices=[
            "full", "no_progressive", "constant", "no_conflict_aware",
            "no_adaptive", "no_jerk", "minimal", "very_conservative",
            "conservative", "very_aggressive",
        ],
    )
    parser.add_argument(
        "--legacy_joint_conflict_guidance",
        action="store_true",
        help="Guide both ego and adversary as in the legacy research code",
    )

    args = parser.parse_args()

    cfg = SceneEditingConfig(registered_name=args.registered_name)

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.planner_ckpt_dir is not None:
        cfg.ckpt.planner.ckpt_dir = args.planner_ckpt_dir
        cfg.ckpt.planner.ckpt_key = args.planner_ckpt_key

    if args.predictor_ckpt_dir is not None:
        cfg.ckpt.predictor.ckpt_dir = args.predictor_ckpt_dir
        cfg.ckpt.predictor.ckpt_key = args.predictor_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch
    if args.num_simulation_steps is not None:
        cfg.trajdata.num_simulation_steps = args.num_simulation_steps

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path
        if "nusc_trainval" in cfg.trajdata.trajdata_data_dirs:
            cfg.trajdata.trajdata_data_dirs["nusc_trainval"] = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    # add eval_class into the results_dir
    # cfg.results_dir = os.path.join(cfg.results_dir, cfg.eval_class)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None

    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]
    if "ui" in cfg.edits.editing_source:
        # can only handle one scene with UI
        cfg.num_scenes_per_batch = 1

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }

    cfg["edits"]["heuristic_config"][0].update({"weight": args.w_collision})
    cfg["edits"]["heuristic_config"][1].update({"weight": args.w_offroad})

    cfg.update({"n_step_action": args.n_step_action})
    selected_vehicles = None
    if "conflict" in cfg.edits.editing_source:
        selected_vehicles = load_selected_vehicles(args.selected_vehicles_path)

    cfg.lock()
    run_scene_editor(
        cfg,
        save_cfg=True,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
        part_control=args.part_control,
        controllable_agent=args.controllable_agent,
        num_steps_intervention=args.num_steps_intervention,
        selected_vehicles=selected_vehicles,
        conflict_ablation=args.v3_ablation_variant,
        conflict_adversary_only=not args.legacy_joint_conflict_guidance,
    )
