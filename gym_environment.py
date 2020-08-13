import numpy as np

import gym
from gym import spaces

from vcg_solver import vcg_prioritizer, vcg_allocator, compute_overall_payment, compute_payment

class OneBidderEnv():#gym.Env):
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

    def step(self, action):
        # Execute one time step within the environment
        done = True
        reward = self.calculate_vcg_reward(self.utility_input, action)
        return self.utility_input, reward, done, {}

    def calculate_vcg_reward(self, utility, bids):
        print("New Run")
        utility_matrix = np.cumsum(np.flip(np.sort(utility.reshape((-1, self.config["bids_per_participant"])),axis=1), axis=1), axis=1)
        all_bids = np.concatenate((bids, self.config["rest_of_bids"]))
        print(all_bids)
        full_bids_matrix = np.cumsum(np.flip(np.sort(all_bids.reshape((-1, self.config["bids_per_participant"])), axis=1), axis=1), axis=1)
        print(full_bids_matrix)
        positions = vcg_prioritizer(all_bids, self.config["bids_per_participant"])
        main_allocation = vcg_allocator(positions, self.config["items_to_sell"], self.config["number_of_players"])
        overall_allocation_value = compute_overall_payment(main_allocation, full_bids_matrix)
        payments = compute_payment(positions, full_bids_matrix, range(self.config['players_colluding']),
                                   self.config["items_to_sell"],
                                   self.config["number_of_players"], utility_matrix, main_allocation,
                                   overall_allocation_value)
        return payments

    def reset(self):

        if self.config["distribution_type_colluders"] == "static":
            self.utility_input = self.config["utility"]
        else:
            raise Exception('distribution type colluders not yet implemented')
        if self.config["distribution_type_colluders"] == "static":
            pass  # self.config["rest_of_bids"] will thus not change
        else:
            raise Exception('distribution type other bidders not yet implemented')
        return

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

if __name__ =="__main__":
    rest_of_bids = np.random.uniform(size=(3 * 3, ))
    utility = 2 * np.random.uniform(size=(2 * 3,))

    config = {
        "players_colluding": 2,
        'bids_per_participant': 3,
        "rest_of_bids": rest_of_bids,
        "items_to_sell": 3,
        "number_of_players": 5,
        "distribution_type_colluders": "static",
        "distribution_type": "static",
        "max_count": 3,
        "utility": utility
    }
    env = OneBidderEnv(config)
    z = 0
