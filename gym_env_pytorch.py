import numpy as np

import gym
from gym import spaces

from vcg_solver import vcg_prioritizer, vcg_allocator, compute_overall_payment, compute_payment, vcg_allocator_torch, compute_payment_torch
import torch

MIN_BID_VALUE, MAX_BID_VALUE = 0.0, 1.0


class OneBidderEnv():  # gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config):
        super(OneBidderEnv, self).__init__()
        self.config = config
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.players_colluding = self.config["players_colluding"]
        self.bids_per_participant = self.config["bids_per_participant"]
        self.items_to_sell = self.config["items_to_sell"]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.players_colluding * self.bids_per_participant,),
                                       dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1.0,
                                            shape=(self.players_colluding * self.bids_per_participant,),
                                            dtype=np.float32)
        self.utility_input = config["utility"]
        self.initial_utility = self.utility_input
        self.initial_rest_of_bids = config["rest_of_bids"]
        self.rest_of_bids = self.initial_rest_of_bids

    def step(self, action):
        # Execute one time step within the environment
        done = True
        reward = self.calculate_vcg_reward(self.utility_input, action)
        return self.utility_input, reward, done, {}

    def calculate_vcg_reward(self, utility, bids):
        utility_matrix = torch.cumsum(
            torch.sort(torch.reshape(utility, (-1, self.config["bids_per_participant"])), dim=1, descending=True), dim=1)
        all_bids = torch.cat((bids, self.rest_of_bids), 0)
        full_bids_matrix = torch.cumsum(
            torch.sort(torch.reshape(all_bids, (-1, self.config["bids_per_participant"])), dim=1, descending=True), dim=1)
        positions = vcg_allocator_torch(all_bids, self.config["bids_per_participant"])
        main_allocation = vcg_allocator_torch(positions, self.config["items_to_sell"], self.config["number_of_players"])
        overall_allocation_value = compute_overall_payment(main_allocation, full_bids_matrix)
        payments = compute_payment_torch(positions, full_bids_matrix, range(self.config['players_colluding']),
                                   self.config["items_to_sell"],
                                   self.config["number_of_players"], utility_matrix, main_allocation,
                                   overall_allocation_value)
        return payments

    def reset(self):

        if self.config["distribution_type_reset_outsiders"] == "static":
            self.utility_input = self.initial_utility
        else:
            raise Exception('distribution type colluders not yet implemented')
        if self.config["distribution_type_reset_colluders"] == "static":
            self.rest_of_bids = self.initial_rest_of_bids
        else:
            raise Exception('distribution type other bidders not yet implemented')
        return

    def make_sample_colluders(bidder, size):
        if bidder.config["distribution_type_colluders"] == "uniform":
            bids = np.random.uniform(size=(size, bidder.players_colluding * bidder.bids_per_participant,))
            return np.flip(np.sort(bids, axis=1), axis=1)
        elif bidder.config["distribution_type_outsiders"] == "perturbation":
            std = bidder.config["perturbation_std"]
            perturbation = np.random.normal(loc=1, scale=std,
                                            size=(size, bidder.players_colluding * bidder.bids_per_participant,))
            base_bid = bidder.initial_utility
            value = np.flip(np.sort(np.clip(perturbation * base_bid[None, :], MIN_BID_VALUE, MAX_BID_VALUE), axis=1))
            return value

        else:
            raise Exception("distribution not yet implemented")

    def make_sample_rest_of_players(bidder, size):
        number_of_bidders = bidder.config["number_of_players"] - bidder.players_colluding
        full_size = (size, number_of_bidders * bidder.bids_per_participant,)
        if bidder.config['distribution_type_outsiders'] == "uniform":
            bids = np.random.uniform(size=full_size)
            return np.flip(np.sort(bids, axis=1), axis=1)
        if bidder.config["distribution_type_outsiders"] == "perturbation":
            std = bidder.config["perturbation_std"]
            perturbation = np.random.normal(loc=1, scale=std, size=full_size)
            base_bid = bidder.rest_of_bids
            value = np.flip(np.sort(np.clip(perturbation * base_bid[None, :], MIN_BID_VALUE, MAX_BID_VALUE), axis=1))
            return value

        else:
            raise Exception("distribution not yet implemented")

    def resample(self):
        self.utility_input = self.make_sample_colluders(1)[0]
        self.rest_of_bids = self.make_sample_rest_of_players(1)[0]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return


if __name__ == "__main__":
    rest_of_bids = np.random.uniform(size=(3 * 3,))
    utility = 2 * np.random.uniform(size=(2 * 3,))

    config = {
        "players_colluding": 2,
        'bids_per_participant': 3,
        "rest_of_bids": rest_of_bids,
        "items_to_sell": 3,
        "number_of_players": 5,
        "distribution_type_reset_outsiders": "static",
        "distribution_type_reset_colluders": "static",
        "distribution_type_colluders": "uniform",
        "distribution_type_outsiders": "uniform",
        "max_count": 3,
        "utility": utility,
        "perturbation_std": 0.1
    }
    env = OneBidderEnv(config)
    z = 0
