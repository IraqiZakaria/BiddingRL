import tensorflow as tf
import numpy as np


def vcg_allocation(colluders_bids: tf.Tensor, rest_of_bidders_bids: tf.Tensor, number_of_items_per_player: int,
                   number_of_items_to_sell: int):
    sort_axis = -1
    args_colluders = tf.argsort(colluders_bids, axis=sort_axis, direction="ASCENDING", stable=False, name=None)
    args_others = tf.argsort(rest_of_bidders_bids, axis=sort_axis, direction="ASCENDING", stable=False, name=None)
    all_bids = tf.concat([colluders_bids, rest_of_bidders_bids], axis=sort_axis)
    bids_allowed = tf.argsort(all_bids, axis=sort_axis, direction="ASCENDING", stable=False, name=None)
    positions = tf.math.floordiv(bids_allowed,
                                 number_of_items_per_player)  # Get the ordeed index of the bidders to,prioritize
    realized_bids = positions[:min(number_of_items_to_sell, tf.shape(positions))]


def vcg_prioritizer(full_bids_list, number_of_items_per_player: int):
    sort_axis = -1
    bids_allowed = np.flip(np.argsort(full_bids_list, axis=sort_axis), sort_axis)
    positions = np.floor_divide(bids_allowed,
                                number_of_items_per_player)  # Get the ordeed index of the bidders to,prioritize
    return positions


def vcg_allocator(positions: tf.Tensor, number_of_items_to_sell: int, number_of_players: int):
    realized_bids = positions[:number_of_items_to_sell]
    counters = np.bincount(realized_bids, minlength=number_of_players)
    return counters


def vcg_allocator_bidder(positions: tf.Tensor, bidder: int, number_of_items_to_sell: int, number_of_players: int):
    capped_positions = []
    for element in positions:
        if element != bidder:
            capped_positions.append(element)
    new_positions = np.array(capped_positions)
    return vcg_allocator(new_positions, number_of_items_to_sell, number_of_players)


def compute_payment(positions, full_bids_matrix, bidders_to_compute, number_of_items_to_sell: int,
                    number_of_players: int, utility_matrix, main_allocation, overall_allocation_value):
    any_bidder = False
    for element in bidders_to_compute:
        if main_allocation[element] != 0:
            any_bidder = True
            break
    if any_bidder == False:
        return np.zeros(shape=(len(bidders_to_compute)))
    valuation = []
    for bidder in bidders_to_compute:
        print("bidder" + str(bidder) +" :" )
        if main_allocation[bidder] == 0:
            print("pass bidder")
            valuation.append(0.0)
        else:
            new_allocation = vcg_allocator_bidder(positions, bidder, number_of_items_to_sell, number_of_players)
            print(new_allocation)
            payment = overall_allocation_value - compute_overall_payment(new_allocation, full_bids_matrix)
            if payment<0:
                print("overall allocation value")
                print(overall_allocation_value)
                print("allocation_bidder")
                print(compute_overall_payment(new_allocation, full_bids_matrix))
            valuation.append(payment)
    utility_team = compute_utility_team(main_allocation, utility_matrix, bidders_to_compute)
    print(utility_team)
    print(valuation)
    return utility_team - sum(valuation)


def compute_overall_payment(current_allocation, full_bids_matrix):
    sum = 0
    count = -1
    for value in current_allocation:
        count += 1
        if value > 0:
            sum += full_bids_matrix[count, value - 1]
    return sum


def compute_utility_team(current_allocation, utility_matrix, team: list):
    sum = 0
    count = -1
    for value in current_allocation:
        count += 1
        if value > 0 and count in team:
            sum += utility_matrix[count, value - 1]
    return sum