import numpy as np


def vcg_prioritizer(full_bids_list, number_of_items_per_player: int):
    sort_axis = -1
    bids_allowed = np.flip(np.argsort(full_bids_list, axis=sort_axis), sort_axis)
    positions = np.floor_divide(bids_allowed,
                                number_of_items_per_player)  # Get the ordeed index of the bidders to,prioritize
    return positions


def vcg_allocator(positions, number_of_items_to_sell: int, number_of_players: int):
    realized_bids = positions[:number_of_items_to_sell]
    counters = np.bincount(realized_bids, minlength=number_of_players)
    return counters


def vcg_allocator_bidder(positions, bidder: int, number_of_items_to_sell: int, number_of_players: int):
    capped_positions = []
    for element in positions:
        if element != bidder:
            capped_positions.append(element)
    new_positions = np.array(capped_positions)
    return vcg_allocator(new_positions, number_of_items_to_sell, number_of_players)

def compute_payment_separated(positions, full_bids_matrix, bidders_to_compute, number_of_items_to_sell: int,
                    number_of_players: int, utility_matrix, main_allocation, overall_allocation_value):
    any_bidder = False
    for element in bidders_to_compute:
        if main_allocation[element] != 0:
            any_bidder = True
            break
    if any_bidder == False:
        np.float64(0)
    valuation = []
    for bidder in bidders_to_compute:
        if main_allocation[bidder] == 0:
            valuation.append(0.0)
        else:
            new_allocation = vcg_allocator_bidder(positions, bidder, number_of_items_to_sell, number_of_players)
            payment = overall_allocation_value - compute_overall_payment(new_allocation, full_bids_matrix)

            valuation.append(payment)
    utility_team = compute_utility_team(main_allocation, utility_matrix, bidders_to_compute)
    return utility_team - sum(valuation)

def compute_payment_combined(positions, full_bids_matrix, bidders_to_compute, number_of_items_to_sell: int,
                    number_of_players: int, main_allocation, overall_allocation_value):
    any_bidder = False
    for element in bidders_to_compute:
        if main_allocation[element] != 0:
            any_bidder = True
            break
    if any_bidder == False:
        np.float64(0)
    valuation = []
    for bidder in bidders_to_compute:
        if main_allocation[bidder] == 0:
            valuation.append(0.0)
        else:
            new_allocation = vcg_allocator_bidder(positions, bidder, number_of_items_to_sell, number_of_players)
            payment = overall_allocation_value - compute_overall_payment(new_allocation, full_bids_matrix)

            valuation.append(payment)
    return  sum(valuation)

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
    for value in current_allocation[team]:
        count += 1
        if value > 0 and count in range(len(team)):

            sum += utility_matrix[count, value - 1]
    return sum
