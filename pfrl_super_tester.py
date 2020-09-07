from gym_environment import OneBidderEnv
from full_board_environment import MultiAgentsEnv, BidderForMultiAgent

import argparse
import sys
import os

import torch.optim as optim
import gym
from gym import spaces
import numpy as np

import pfrl
from pfrl.agents.dqn import DQN
from pfrl import experiments
from pfrl import explorers
from pfrl import nn as pnn
from pfrl import utils
from pfrl import q_functions
from pfrl import replay_buffers

from pfrl_addons import train_agent_with_evaluation

import torch

class Args():
  def __init__(args):

    args.seed=0
    args.gpu=0
    args.final_exploration_steps=10 ** 4
    args.start_epsilon=1.0
    args.end_epsilon=0.1
    args.noisy_net_sigmat=None
    args.demo=False
    args.load=None
    args.steps = 10 ** 4
    args.prioritized_replay=""
    action="store_true"
    args.replay_start_size=1000
    args.target_update_interval=10 ** 2
    args.target_update_method="hard"
    args.soft_update_tau=1e-2
    args.update_interval=1
    args.eval_n_runs=100
    args.eval_interval=10 ** 4
    args.n_hidden_channels=100
    args.n_hidden_layers=2
    args.gamma=0.99
    args.minibatch_size=None
    args.render_train = ""
    action="store_true"
    args.render_eval = ""
    action="store_true"
    args.monitor=""
    action="store_true"
    args.reward_scale_factor=1e-3
    args.num_envs = 1
    args.noisy_net_sigma =  None
    args.actor_learner = False
    args.outdir = "results\\2"

def main(env, args):
    import logging

    logging.basicConfig(level=logging.INFO)
    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(env=env, idx=0, test=False):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        utils.set_random_seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            utils.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if (args.render_eval and test) or (args.render_train and not test):
            env = pfrl.wrappers.Render(env)
        return env
    obs_spaces_dict = {}
    obs_size_dict = {}
    action_space_dict = {}
    timestep_limits = {}
    q_functions_dicts = {}
    explorers_dict = {}
    optimizers_dict = {}
    agents_dict = {}

    for key in env.overall_config.keys():
        sub_env = env.overall_config[key]
        env.overall_config[key] = make_env(env = sub_env, test= False)
        timestep_limits[key] = sub_env.spec.max_episode_steps
        obs_space = sub_env.observation_space
        obs_spaces_dict[key]  = sub_env.observation_space
        obs_size = obs_space.low.size
        obs_size_dict[key] = obs_size
        action_space = sub_env.action_space
        action_space_dict[key] = sub_env.action_space

        if isinstance(action_space, spaces.Box):
            action_size = action_space.low.size
            # Use NAF to apply DQN to continuous action spaces
            q_func = q_functions.FCQuadraticStateQFunction(
                obs_size,
                action_size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
                action_space=action_space,
            )
            # Use the Ornstein-Uhlenbeck process for exploration
            ou_sigma = (action_space.high - action_space.low) * 0.2
            explorer = explorers.AdditiveOU(sigma=ou_sigma)
        else:
            n_actions = action_space.n
            q_func = q_functions.FCStateQFunctionWithDiscreteAction(
                obs_size,
                n_actions,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
            )
            # Use epsilon-greedy for exploration
            explorer = explorers.LinearDecayEpsilonGreedy(
                args.start_epsilon,
                args.end_epsilon,
                args.final_exploration_steps,
                action_space.sample,
            )
        if args.noisy_net_sigma is not None:
            pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
            # Turn off explorer
            explorer = explorers.Greedy()
        opt = optim.Adam(q_func.parameters())

        rbuf_capacity = 5 * 10 ** 5
        if args.minibatch_size is None:
            args.minibatch_size = 32
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) // args.update_interval
            rbuf = replay_buffers.PrioritizedReplayBuffer(
                rbuf_capacity, betasteps=betasteps
            )
        else:
            rbuf = replay_buffers.ReplayBuffer(rbuf_capacity)

        agent = DQN(
            q_func,
            opt,
            rbuf,
            gpu=args.gpu,
            gamma=args.gamma,
            explorer=explorer,
            replay_start_size=args.replay_start_size,
            target_update_interval=args.target_update_interval,
            update_interval=args.update_interval,
            minibatch_size=args.minibatch_size,
            target_update_method=args.target_update_method,
            soft_update_tau=args.soft_update_tau,
        )

        if args.load:
            agent.load(args.load)

        agents_dict[key] = agent
        q_functions_dicts[key] = q_func
        explorers_dict[key] = explorer
        optimizers_dict[key] = opt

    # eval_env = make_env( test=True)
    eval_env = None

    train_agent_with_evaluation(
        agent=agents_dict,
        env=env,
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        outdir=args.outdir,
        eval_env=eval_env,
        train_max_episode_len=timestep_limits,
    )



if __name__ == "__main__":
    args = Args()

    utility_add_on = 1
    bids_per_participant = 3
    items_to_sell = 3
    number_of_collusions = 3
    std = 0.2
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
            "distribution_type_reset_colluders": "perturbation",
            "perturbation_std_colluders" : std,
            "utility_type": "separated",
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
    actions = single_env.reset()

    print(single_env.step(actions))
    main(single_env, args)