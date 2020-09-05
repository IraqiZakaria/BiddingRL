import numpy as np

from vcg_solver import vcg_allocator, vcg_prioritizer, compute_overall_payment, compute_payment_separated, \
    compute_payment_combined
from gym import spaces

from gym_environment import OneBidderEnv

MIN_BID_VALUE, MAX_BID_VALUE = 0.0, 1.0
MAX_REWARD = 20


class Spec():
    max_episode_steps = 1


class BidderForMultiAgent():
    def __init__(self, config):
        self.config = config
        self.utility_input = config["utility"]
        self.initial_utility = self.utility_input

        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(len(self.config["players_colluding"]) * self.config["bids_per_participant"],),
                                       dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1.0,
                                            shape=(
                                                len(self.config["players_colluding"]) * self.config["bids_per_participant"],),
                                            dtype=np.float32)
        self.reward_range = (0, MAX_REWARD)
        self.metadata = {}
        self.spec = Spec()
        # config utility
        # config['players_colluding']
        # bids_per_participant have to be the same for everyone

    def reset(self):
        if self.config["distribution_type_reset_colluders"] == "static":
            pass
        elif self.config["distribution_type_reset_colluders"] == "perturbation":
            self.utility_input = self.make_sample_colluders(1)[0]
        else:
            raise Exception('distribution type other bidders not yet implemented')
        return self.utility_input

    def make_sample_colluders(self, size):
        if self.config["distribution_type_reset_colluders"] == "uniform":
            bids = np.random.uniform(
                size=(size, len(self.config.players_colluding) * self.config["bids_per_participant"],))
            return np.flip(np.sort(bids, axis=1), axis=1)
        elif self.config["distribution_type_reset_colluders"] == "perturbation":
            std = self.config["perturbation_std_colluders"]
            perturbation = np.random.normal(loc=1, scale=std,
                                            size=(size, len(
                                                self.config["players_colluding"]) * self.config["bids_per_participant"],))
            base_bid = self.initial_utility
            value = np.flip(np.sort(np.clip(perturbation * base_bid[None, :], MIN_BID_VALUE, MAX_BID_VALUE), axis=1))
            return value

        else:
            raise Exception("distribution not yet implemented")


class MultiAgentsEnv():
    def __init__(self, dict_of_colluders_configs, parameters):
        # Do a better init :  an init that will need only one utility input, and
        self.overall_config = dict_of_colluders_configs
        self.parameters = parameters

    def reset(self):
        # All the other bidders maps will be static btw
        # Since this one is not creating any rest of bids step, and will only be updating the tribes revenue, no need to act
        # in an inner way between the agents
        all_bids = {}
        for element in self.overall_config.keys():
            all_bids[element] = self.overall_config[element].reset()
        return all_bids

    def step(self, actions):
        # I will do stategic combined and separated utilities separately
        # actions are here a dict again, and will be the key to how we will effectively step in our model.
        new_obs, rewards, dones, infos = {}, {}, {}, {}
        dones['__all__'] = True

        indexes = {}
        bids = []
        for key in actions.keys():
            indexes[key] = self.overall_config[key].config['players_colluding']
            bids.append(actions[key])
        bids = np.concatenate(bids)
        full_bids_matrix = np.cumsum(
            np.flip(np.sort(bids.reshape((-1, self.parameters["bids_per_participant"])), axis=1), axis=1), axis=1)
        positions = vcg_prioritizer(bids, self.parameters["bids_per_participant"])

        main_allocation = vcg_allocator(positions, self.parameters["items_to_sell"],
                                        self.parameters["number_of_players"])
        overall_allocation_value = compute_overall_payment(main_allocation, full_bids_matrix)
        for key in actions.keys():
            utility = self.overall_config[key].utility_input

            dones[key] = True
            new_obs[key] = utility
            if self.overall_config[key].config["utility_type"] == "separated":
                utility_matrix = np.cumsum(
                    np.flip(np.sort(utility.reshape((-1, self.parameters["bids_per_participant"])), axis=1), axis=1),
                    axis=1)
                payments = compute_payment_separated(positions, full_bids_matrix, indexes[key],
                                                     self.parameters["items_to_sell"],
                                                     self.parameters["number_of_players"], utility_matrix,
                                                     main_allocation,
                                                     overall_allocation_value)
                rewards[key] = payments

            elif self.overall_config[key].config["utility_type"] == "combined":
                utility_matrix = np.cumsum(np.flip(np.sort(utility)))

                revenue = utility_matrix[np.sum(main_allocation[indexes[key]]) - 1]
                payments = compute_payment_combined(positions, full_bids_matrix,
                                                    indexes[key],
                                                    self.parameters["items_to_sell"],
                                                    self.parameters["number_of_players"], main_allocation,
                                                    overall_allocation_value)
                rewards[key] = revenue - payments
            else:
                raise Exception("Method not yet implemented")
        return new_obs, rewards, dones, infos


if __name__ == "__main__":
    utility_add_on = 1
    bids_per_participant = 3
    items_to_sell = 3
    number_of_collusions = 3
    dict_of_colluders_configs = {}
    for k in range(number_of_collusions):
        players_colluding = range(2*k, 2*k + 2)
        number_of_players_colluding = len(players_colluding)
        utility = utility_add_on * np.flip(
            np.sort(np.random.uniform(size=(number_of_players_colluding * bids_per_participant,)), axis=-1))

        config = {
            "players_colluding" : players_colluding,
            "bids_per_participant":  bids_per_participant,
            "utility": utility,
            "distribution_type_reset_colluders": "static",
            "utility_type": "separated"
        }
        dict_of_colluders_configs["colluders_"+str(k)] = BidderForMultiAgent(config)
    parameters = {
        "number_of_players": bids_per_participant * number_of_collusions * 2,
        "items_to_sell": items_to_sell,
        "bids_per_participant": bids_per_participant
    }
    env = MultiAgentsEnv(dict_of_colluders_configs, parameters)
    env.reset()
