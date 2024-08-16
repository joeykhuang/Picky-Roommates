import pandas as pd
import itertools
from tqdm import trange
import time
from preferences import generate_preferences, calculate_utilities
from greedy import generate_greedy_utilities
from mwis import *
from ef import *


# Generate all possible partitions for m tenants and n rooms
def generate_partitions(m, n):
    def recursive_assign(assigned, remaining):
        if not remaining:
            # Only yield when all rooms have at least one person
            if len(assigned) == n and all(assigned):
                yield assigned
            return
        # Take the next person to assign
        person = remaining[0]
        new_remaining = remaining[1:]
        for i in range(len(assigned)):
            if assigned[i] and len(assigned[i]) < 2:  # Add person to existing room if it has less than 2 people
                new_assigned = [list(room) for room in assigned]
                new_assigned[i].append(person)
                yield from recursive_assign(new_assigned, new_remaining)
        if len(assigned) < n:  # Start a new room if not all rooms are used
            yield from recursive_assign(assigned + [[person]], new_remaining)

    if m < n:
        print("Error: m must be greater than or equal to n")
        return

    partitions = list(recursive_assign([], list(range(1, m + 1))))
    partitions_new = []
    for partition in partitions:
        partitions_new.extend(list(itertools.permutations(partition)))
    return partitions_new

# Simulates the greedy and greedy with bipartite matching solutions
def simulate(m, n, num_simulations, singles_preference_param):
    greedy_results = []
    greedy_match_results = []
    optimal_results = []
    for _ in range(num_simulations):
        # generate random preferences
        prefs = generate_preferences(m, n, singles_preference_param)

        # generate greedy and greedy with matching solutions
        greedy_utilities_df = generate_greedy_utilities(prefs)
        greedy_assignment_partition, greedy_assignment_result = create_greedy_partition(m, n, greedy_utilities_df)
        greedy_match_partition, greedy_match_result = greedy_match(greedy_assignment_partition, greedy_utilities_df)

        # enumerate all possible partitions to find the optimal assignment
        optimal_assignments = calculate_utilities(generate_partitions(m, n), prefs)
        optimal_assignment_result = pd.DataFrame(optimal_assignments)[1].max()

        greedy_results.append(greedy_assignment_result)
        greedy_match_results.append(greedy_match_result)
        optimal_results.append(optimal_assignment_result)

    return greedy_results, greedy_match_results, optimal_results

# Simulate the greedy and greedy with bipartite matching and MWIS solutions
def simulate_mwis(m, n, num_simulations, do_optimal=True):
    greedy_results = []
    greedy_match_results = []
    mwis_results = []
    optimal_results = []
    for _ in range(num_simulations):
        # generate random preferences with singles preference (alpha) fixed at 0.5
        prefs = generate_preferences(m, n, 0.5)
        greedy_utilities_df = generate_greedy_utilities(prefs)
        greedy_assignment_partition, greedy_assignment_result = create_greedy_partition(m, n, greedy_utilities_df)
        greedy_match_partition, greedy_match_result = greedy_match(greedy_assignment_partition, greedy_utilities_df)

        mwis_partition, mwis_result = generate_mwis_solution(m, n, greedy_utilities_df)

        greedy_results.append(greedy_assignment_result)
        greedy_match_results.append(greedy_match_result)
        mwis_results.append(mwis_result)

        if do_optimal:
            optimal_assignments = calculate_utilities(generate_partitions(m, n), prefs)
            optimal_assignment_result = pd.DataFrame(optimal_assignments)[1].max()
            optimal_results.append(optimal_assignment_result)

    return greedy_results, greedy_match_results, mwis_results, optimal_results

# Simuate the MWIS runtimes
def runtime_sim(nmax=12, niter=10):
    timings = []
    for n in trange(2, nmax):
        t = 0
        m = 2 * n
        for _ in range(niter):
            prefs = generate_preferences(m, n, 0.5)
            greedy_utilities_df = generate_greedy_utilities(prefs)
            start = time.time()
            generate_mwis_solution(m, n, greedy_utilities_df)
            end = time.time()
            t += end - start
        timings.append(t / niter)
    return timings

# Simulate MWIS with ghost preferences (m < 2n)
def simulate_mwis_with_ghosts(m, n, num_simulations, singles_preference_param, do_optimal=True):
    greedy_results = []
    greedy_match_results = []
    mwis_results = []
    optimal_results = []
    new_m = 2 * n
    for _ in range(num_simulations):
        prefs = generate_preferences(m, n, singles_preference_param)
        greedy_utilities_df = generate_greedy_utilities(prefs)

        greedy_assignment_partition, greedy_assignment_result = create_greedy_partition(m, n, greedy_utilities_df)
        greedy_match_partition, greedy_match_result = greedy_match(greedy_assignment_partition, greedy_utilities_df)

        if m == 2 * n:
            mwis_partition_ghost, mwis_result = generate_mwis_solution(m, n, generate_greedy_utilities(prefs))
        else:
            possible_mwis_res = []
            prefs_ghosts = add_preferences_with_ghosts(m, n, prefs)
            greedy_utilities_ghosts_df = generate_greedy_utilities(prefs_ghosts)
            mwis_partition_ghost, mwis_result_ghost = generate_mwis_solution(new_m, n, greedy_utilities_ghosts_df)
            possible_mwis_res.append(mwis_result_ghost)

            mwis_result = max(possible_mwis_res)

        greedy_results.append(greedy_assignment_result)
        greedy_match_results.append(greedy_match_result)
        mwis_results.append(mwis_result)

        if do_optimal:
            optimal_assignments = calculate_utilities(generate_partitions(m, n), prefs)
            optimal_assignment_result = pd.DataFrame(optimal_assignments)[1].max()
            optimal_results.append(optimal_assignment_result)

    return greedy_results, greedy_match_results, mwis_results, optimal_results