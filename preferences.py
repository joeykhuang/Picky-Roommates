import numpy as np
from sklearn.preprocessing import normalize

# Generate random preferences from a uniform distribution
def generate_preferences(m, n, singles_preference_param):
    preference_matrix = []
    for i in range(m):
        # generate row-normalized preference matrices
        player_preference = normalize(np.random.rand(m, n), axis=1, norm='l1')
        #player_preference = np.random.rand(m, n)

        # most people would prefer singles, so give singles higher preferences
        player_preference[i] = [x + singles_preference_param for x in player_preference[i]]
        preference_matrix.append(player_preference)

    return preference_matrix

# Calculates the utilities matrix based on a list of partitions and the preferences matrix.
# Used for brute-forcing all possible partitions and finding the max-utility partition
def calculate_utilities(partitions, preferences):
    def calculate_utilities_for_partition(partition, preferences):
        total_utility = 0
        for i in range(len(partition)):
            room = partition[i]
            # singles
            if (len(room) == 1):
                person = room[0] - 1
                total_utility += preferences[person][person][i]

            # doubles
            else:
                person1 = room[0] - 1
                person2 = room[1] - 1
                total_utility += preferences[person1][person2][i] + preferences[person2][person1][i]
        return total_utility

    partition_utilities = []
    for partition in partitions:
        partition_utilities.append((partition, calculate_utilities_for_partition(partition, preferences)))
    return partition_utilities