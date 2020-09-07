import logging
import os
from pfrl.experiments.evaluator import Evaluator
import numpy as np


def save_agent(agent, t, outdir, logger, suffix=""):
    dirname = os.path.join(outdir, "{}{}".format(t, suffix))
    for key in agent.keys():
        agent[key].save(dirname + str(key))
    logger.info("Saved the agent to %s", dirname)


def train_agent(
        agents,
        env,
        steps,
        outdir,
        checkpoint_freq=None,
        max_episode_len=None,
        step_offset=0,
        evaluator=None,
        successful_score=None,
        step_hooks=(),
        logger=None,
):
    logger = logger or logging.getLogger(__name__)

    episode_r = {key: 0 for key in agents.keys()}
    episode_r_0 = {key: 0 for key in agents.keys()}
    episode_idx = 0

    # o_0, r_0
    obs = env.reset()

    t = step_offset
    for key in agents.keys():
        if hasattr(agents[key], "t"):
            agents[key].t = step_offset

    episode_len = 0
    try:
        while t < steps:
            actions = {}
            for key in agents.keys():
                actions[key] = np.clip(agents[key].act(obs[key]), a_min=0, a_max=1)
            # a_t

            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(actions)
            t += 1
            episode_r = r
            episode_len += 1
            # TODO
            maximum_episode_len = max(max_episode_len.values())
            reset = {key: episode_len == max_episode_len[key] or info[key].get("needs_reset", False) for key in
                     agents.keys()}
            resets = all(value == True for value in reset.values())
            dones = all(value == True for value in done.values())

            for key in agents.keys():
                if t == 1000:
                    a = 0
                agents[key].observe(obs[key], r[key], done[key], reset[key])

            for hook in step_hooks:
                hook(env, agents, t)

            if dones or resets or t == steps:
                logger.info(
                    "outdir:%s step:%s episode:%s R:%s",
                    outdir,
                    t,
                    episode_idx,
                    episode_r,
                )
                for key in agents.keys():
                    logger.info("statistics:%s %s", key,  agents[key].get_statistics())
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(t=t, episodes=episode_idx + 1)
                    if (
                            successful_score is not None
                            and evaluator.max_score >= successful_score
                    ):
                        break
                if t == steps:
                    break
                # Start a new episode
                episode_r = episode_r_0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agents, t, outdir, logger, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agents, t, outdir, logger, suffix="_except")
        raise

    # Save the final model
    save_agent(agents, t, outdir, logger, suffix="_finish")


def train_agent_with_evaluation(
        agent,
        env,
        steps,
        eval_n_steps,
        eval_n_episodes,
        eval_interval,
        outdir,
        checkpoint_freq=None,
        train_max_episode_len=None,
        step_offset=0,
        eval_max_episode_len=None,
        eval_env=None,
        successful_score=None,
        step_hooks=(),
        save_best_so_far_agent=True,
        use_tensorboard=False,
        logger=None,
):
    """Train an agent while periodically evaluating it.

    Args:
        agent: A pfrl.agent.Agent
        env: Environment train the agent against.
        steps (int): Total number of timesteps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_episodes (int): Number of episodes at each evaluation phase.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output data.
        checkpoint_freq (int): frequency at which agents are stored.
        train_max_episode_len (int): Maximum episode length during training.
        step_offset (int): Time step from which training starts.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If None, train_max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            than or equal to this value if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation
            phase, if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len

    evaluator = None
    # evaluator = Evaluator(
    #     n_steps=eval_n_steps,
    #     n_episodes=eval_n_episodes,
    #     eval_interval=eval_interval,
    #     outdir=outdir,
    #     max_episode_len=eval_max_episode_len,
    #     step_offset=step_offset,
    #     save_best_so_far_agent=save_best_so_far_agent,
    #     use_tensorboard=use_tensorboard,
    #     logger=logger,
    #     agent=agent,
    #     env=eval_env,
    # )

    train_agent(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger,
    )
