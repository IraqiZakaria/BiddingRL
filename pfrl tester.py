"""An example of training DQN against OpenAI Gym Envs.
This script is an example of training a DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.
To solve CartPole-v0, run:
    python train_dqn_gym.py --env CartPole-v0
To solve Pendulum-v0, run:
    python train_dqn_gym.py --env Pendulum-v0
"""
from gym_environment import OneBidderEnv

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

import torch


def main(env):
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--final-exploration-steps", type=int, default=10 ** 4)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.1)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10 ** 4)
    parser.add_argument("--prioritized-replay", action="store_true")
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=10 ** 2)
    parser.add_argument("--target-update-method", type=str, default="hard")
    parser.add_argument("--soft-update-tau", type=float, default=1e-2)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10 ** 4)
    parser.add_argument("--n-hidden-channels", type=int, default=100)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--reward-scale-factor", type=float, default=1e-3)
    parser.add_argument(
        "--actor-learner",
        action="store_true",
        help="Enable asynchronous sampling with asynchronous actor(s)",
    )  # NOQA
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help=(
            "The number of environments for sampling (only effective with"
            " --actor-learner enabled)"
        ),
    )  # NOQA
    args = parser.parse_args()

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
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

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

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

    eval_env = make_env( test=True)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )

    elif not args.actor_learner:

        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            eval_env=eval_env,
            train_max_episode_len=timestep_limit,
        )
    else:
        # using impala mode when given num of envs

        # When we use multiple envs, it is critical to ensure each env
        # can occupy a CPU core to get the best performance.
        # Therefore, we need to prevent potential CPU over-provision caused by
        # multi-threading in Openmp and Numpy.
        # Disable the multi-threading on Openmp and Numpy.
        os.environ["OMP_NUM_THREADS"] = "1"  # NOQA

        (
            make_actor,
            learner,
            poller,
            exception_event,
        ) = agent.setup_actor_learner_training(args.num_envs)

        poller.start()
        learner.start()

        experiments.train_agent_async(
            processes=args.num_envs,
            make_agent=make_actor,
            make_env=make_env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            stop_event=learner.stop_event,
            exception_event=exception_event,
        )

        poller.stop()
        learner.stop()
        poller.join()
        learner.join()


if __name__ == "__main__":
    number_of_players_colluding = 2
    other_bidders = 3
    allowed_bids = 3
    items_to_sell = 4
    utility_add_on = 1
    std_rest_of_bidders = 0.2
    std_colluders = 0.1
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    rest_of_bids = np.flip(np.sort(np.random.uniform(size=(other_bidders * allowed_bids,))))
    utility = utility_add_on * np.flip(np.sort(np.random.uniform(size=(number_of_players_colluding * allowed_bids,)), axis=-1))
    config = {
        "players_colluding": number_of_players_colluding,
        'bids_per_participant': allowed_bids,
        "rest_of_bids": rest_of_bids,
        "items_to_sell": items_to_sell,
        "number_of_players": number_of_players_colluding + other_bidders,
        "distribution_type_reset_outsiders": "perturbation",
        "distribution_type_reset_colluders": "perturbation",
        "max_count": allowed_bids,
        "utility": utility,
        "perturbation_std_colluders": std_colluders,
        "perturbation_std_rest_of_bidders": std_rest_of_bidders
    }
    env = OneBidderEnv(config)
    main(env)
    print(utility)
    print(rest_of_bids)