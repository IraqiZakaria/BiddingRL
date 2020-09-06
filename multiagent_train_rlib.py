from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import gym
import ray
from ray import tune

from full_board_environment import MultiAgentsEnv, BidderForMultiAgent
from ray.tune.registry import register_env

if __name__ == "__main__":
    utility_add_on = 1
    bids_per_participant = 3
    items_to_sell = 3
    number_of_collusions = 3
    dict_of_colluders_configs = {}
    for k in range(number_of_collusions):
        players_colluding = range(2 * k, 2 * k + 2)
        number_of_players_colluding = len(players_colluding)
        utility = utility_add_on * np.flip(
            np.sort(np.random.uniform(size=(number_of_players_colluding * bids_per_participant,)), axis=-1))

        config = {
            "players_colluding": players_colluding,
            "bids_per_participant": bids_per_participant,
            "utility": utility,
            "distribution_type_reset_colluders": "static",
            "utility_type": "separated"
        }
        dict_of_colluders_configs["colluders_" + str(k)] = BidderForMultiAgent(config)
    parameters = {
        "number_of_agents": number_of_collusions,
        "number_of_players": number_of_collusions * 2,
        "items_to_sell": items_to_sell,
        "bids_per_participant": bids_per_participant
    }


    def env_creator(_):
        return MultiAgentsEnv(dict_of_colluders_configs, parameters)

    single_env = MultiAgentsEnv(dict_of_colluders_configs, parameters)
    env_name = "Board"

    register_env(env_name, env_creator)

    policy_graphs = {}
    for key in single_env.overall_config.keys():
        policy_graphs[key] = single_env.overall_config[key].gen_policy()
    policy_ids = list(policy_graphs.keys())
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--num-policies', type=int, default=3)
    parser.add_argument('--num-iters', type=int, default=100000)
    parser.add_argument('--simple', action='store_true')
    args = parser.parse_args()

    tune.run(
        'PPO',
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=50,
        config={
            'env': 'Board',
            'lambda': 0.95,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'vf_clip_param': 10.0,
            'entropy_coeff': 0.01,
            'train_batch_size': 2000,
            'sample_batch_size': 100,
            'sgd_minibatch_size': 500,
            'num_sgd_iter': 10,
            'num_workers': 4,
            'num_envs_per_worker': 1,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'vf_share_layers': 'true',
            'num_gpus': 0,
            'lr': 2.5e-4,
            'log_level': 'DEBUG',
            'simple_optimizer': args.simple,
            'multiagent': {
                'policies': policy_graphs,
                'policy_mapping_fn': tune.function(
                    lambda agent_id: policy_ids[int(agent_id[6:])]),
            },
        },
    )