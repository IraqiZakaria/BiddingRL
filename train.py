from models import DQN
from replayMemory import ReplayMemory
import torch.optim as optim
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
from replayMemory import Transition
import torch.nn.functional as F


from gym_environment import OneBidderEnv

torch.set_default_tensor_type('torch.cuda.FloatTensor')
rest_of_bids = np.flip(np.sort(np.random.uniform(size=(3 * 3,))))
utility = 1.2 * np.flip(np.sort(np.random.uniform(size=(2 * 3,)), axis=-1))

config = {
    "players_colluding": 2,
    'bids_per_participant': 3,
    "rest_of_bids": rest_of_bids,
    "items_to_sell": 3,
    "number_of_players": 5,
    "distribution_type_reset_outsiders": "static",
    "distribution_type_reset_colluders": "static",
    "distribution_type_colluders": "perturbation",
    "distribution_type_outsiders": "perturbation",
    "max_count": 3,
    "utility": utility,
    "perturbation_std": 0.1
}
env = OneBidderEnv(config)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

in_channels = env.players_colluding * env.bids_per_participant
n_actions = env.players_colluding * env.bids_per_participant
device = "cuda"
policy_net = DQN(in_channels=in_channels, n_actions=n_actions).to(device)
target_net = DQN(in_channels=in_channels, n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state)
    else:
        return torch.tensor(np.random.uniform(size=(n_actions, )), device=device, dtype=torch.long)


episode_durations = []
rewards = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.cpu().numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.cpu().numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    is_ipython = 'inline' in matplotlib.get_backend()

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_rewards():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.cpu().numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(rewards)

    plt.pause(0.001)  # pause a bit so that plots are updated
    is_ipython = 'inline' in matplotlib.get_backend()

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)

    #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                      batch.next_state)), device=device, dtype=torch.bool)
    #non_final_next_states = torch.cat([s for s in batch.next_state
    #                                            if s is not None])

    # state_batch = torch.cat(batch.state, dim = 0)
    states = [element[None, :] for element in batch.state]
    state_batch = torch.cat(states, dim=-2)
    actions = [element[None, :] for element in batch.action]
    # Initial action_batch = torch.cat(batch.action, dim=1)
    action_batch = torch.cat(actions, dim=-2)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = policy_net(state_batch)


    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    #next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = reward_batch
    print(reward_batch)
    # This is supposed to give, in our case, only the reward to add

    # Compute Huber loss
    zeros = torch.zeros(BATCH_SIZE, device=device)
    loss = - F.smooth_l1_loss(expected_state_action_values.unsqueeze(1), zeros)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    #env.resample()
    state = env.utility_input
    state = torch.from_numpy(state.copy()).float().cpu()
    for t in range(100):
        # Select and perform an action
        action = select_action(state).cpu()
        _, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device=device)

        if not done:
            a = 0
            # Not implemented and not needed in our case
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            rewards.append(list(reward.cpu().numpy())[0])
            episode_durations.append(t + 1)
            plot_rewards()
            #plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()
plt.show()
