import numpy as np
import pandas as pd
import itertools
import scipy.optimize


# Generate the greedy utilites matrix used by both greedy and MWIS algorithms
def generate_greedy_utilities(preferences):
    m = len(preferences)
    n = len(preferences[0][0])

    # generate all possible pairings between tenants and roommate groups with rooms
    possible_pairings = list(itertools.combinations_with_replacement(range(1, m + 1), 2))
    possible_pairings_with_rooms = itertools.product(possible_pairings, range(n))

    # calculate the utilities for each pairing
    rooms_and_utiliites = []
    for pairing in possible_pairings_with_rooms:
        room = pairing[1]
        person1 = pairing[0][0] - 1
        person2 = pairing[0][1] - 1
        if person1 == person2:
            utility = preferences[person1][person1][room]
        else:
            utility = preferences[person1][person2][room] + preferences[person2][person1][room]

        rooms_and_utiliites.append([pairing, utility])

    df = pd.DataFrame(rooms_and_utiliites, columns = ["partition", "utility"])
    df[['people', 'room']] = pd.DataFrame(df['partition'].tolist(), index=df.index)
    df[['person1', 'person2']] = pd.DataFrame(df['people'].tolist(), index=df.index)
    df = df.drop(["partition", "people"], axis=1)

    return df

# Creates the greedy partition by repeatedly taking the top utility pairing from the greedy matrix
def create_greedy_partition(m, n, greedy_df):
    total_utility = 0
    partitions = []
    num_people_sorted = 0
    num_rooms_sorted = 0
    
    # continue while the dataframe is non-empty
    while len(greedy_df) > 0:
        m_prime = m - num_people_sorted
        n_prime = n - num_rooms_sorted
        if m_prime == 2 * n_prime:
            greedy_df = greedy_df[(greedy_df["person1"] != greedy_df["person2"])]
        if m_prime == n_prime:
            greedy_df = greedy_df[(greedy_df["person1"] == greedy_df["person2"])]
        max_partition_idx = greedy_df['utility'].idxmax()
        max_partition = greedy_df.loc[max_partition_idx]
        max_partition_p1, max_partition_p2, max_partition_rm = [max_partition["person1"], max_partition["person2"], max_partition["room"]]
        if max_partition_p1 == max_partition_p2 and m_prime - 1 > 2 * (n_prime - 1):
            greedy_df = greedy_df[(greedy_df["person1"] != greedy_df["person2"])]
            continue
        partitions.append([[max_partition_p1, max_partition_p2, max_partition_rm], max_partition["utility"]])
        greedy_df = greedy_df[(greedy_df["room"] != max_partition_rm) &
        (greedy_df["person1"] != max_partition_p1) & (greedy_df["person2"] != max_partition_p2) &
        (greedy_df["person1"] != max_partition_p2) & (greedy_df["person2"] != max_partition_p1)]
        num_people_sorted += int(max_partition_p1 != max_partition_p2) + 1
        num_rooms_sorted += 1
        total_utility += max_partition["utility"]

    return [partitions, total_utility]

# Applies the bipartite matching algorithm for roomming groups generated by the greedy algorithm to rooms
def greedy_match(greedy_partitions, greedy_df):
    possible_pairs = []
    cost_matrix = []
    for pair, cost in greedy_partitions:
        p1, p2, _ = pair
        possible_pairs.append((p1, p2) if p1 < p2 else (p2, p1))
    for _, row in greedy_df.iterrows():
        p1, p2 = row["person1"], row["person2"]
        row_tuple = (p1, p2) if p1 < p2 else (p2, p1)
        if row_tuple in possible_pairs:
            cost_matrix.append([row_tuple, row["room"], -row["utility"]])
    cost_df = pd.DataFrame(cost_matrix, columns = ["pair", "room", "utility"])
    cost_df = cost_df.pivot(index='pair', columns='room', values='utility')

    # use the linear_sum_assignment function to do bipartite matching on the cost dataframe
    ri, ci = scipy.optimize.linear_sum_assignment(cost_df)
    total_utility = -np.array(cost_df)[ri, ci].sum()
    max_welfare_partitions = []
    for i in range(len(ri)):
        p1, p2 = cost_df.index[ri[i]]
        rm = ci[i]
        max_welfare_partitions.append([p1, p2, rm])
    return max_welfare_partitions, total_utility