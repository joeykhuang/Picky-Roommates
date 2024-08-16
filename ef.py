from mwis import generate_mwis_solution
from greedy import create_greedy_partition, greedy_match, generate_greedy_utilities
from preferences import generate_preferences
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from sklearn.preprocessing import normalize

# Creates the envy matrix: envy_table[i][j] is the amount of utility that i would get 
# from rooming with j in j's assigned room. Also creates dictionaries mapping tenants 
# to their roommates and rooms for easier downstream manipulation.
def get_ef_df(m, partition, prefs):
    person_room_table = {}
    person_rm_table = {}
    envy_table = []

    # create the tenant-roommate and tenant-room mappings
    for part in partition:
        p1, p2, r = part
        person_room_table.update({p1: r, p2: r})
        person_rm_table.update({p1 - 1: p2 - 1, p2 - 1: p1 - 1})

    # find the utility of rooming with j in j's room (switching with j's roommate)
    for i in range(1, m + 1):
        i_utilities = []
        for j in range(1, m + 1):
            j_room = int(person_room_table[j])
            roommate_utility = prefs[i - 1][j - 1][j_room]
            i_utilities.append(roommate_utility)
        envy_table.append(i_utilities)

    envy_df = pd.DataFrame(envy_table)
    return person_rm_table, person_room_table, envy_df

# Finds the EF price vector for different algorithms: 
#   - inv_epsilon != None fixes the epsilon (maximum envy bound) and will generate a maximin price vector for a given assignment (if found)
#   - inv_epsilon == None will minimize epsilon and will find a solution with minimum max envy between any two tenants
#          - share_rent_equally == True will give a equal-rent share price vector where two roommates in a room split the room price regardless of their preferences
#          - share_rent_equally == False will give a non-equal-rent share price vector for roommates where original preferences matter more
def find_ef_price(m, person_rm_table, envy_df, inv_epsilon = None, share_rent_equally=False):
    if inv_epsilon != None:
        prices = range(m)
        model = gp.Model()
        p = model.addVars(prices, vtype=GRB.CONTINUOUS, name="p")
        
        # R is the minimum rent that any tenant pays
        R = model.addVar(vtype=GRB.CONTINUOUS, name="r")
        model.addConstr(p.sum() == 1)
        
        # normalize the new envy dataframe with l1 normalization
        new_envy_df = pd.DataFrame(normalize(envy_df, axis=1, norm='l1'))
        for i in range(m):
            for j in range(m):
                i_rm = int(person_rm_table[i])
                j_rm = int(person_rm_table[j])
                # do not add constraints between roommates
                if j != i_rm and i != j_rm:
                    # epsilon-EF constraint
                    model.addConstr(envy_df.iloc[i, i_rm] - p[i] >= inv_epsilon * (envy_df.iloc[i, j] - p[j]))
                    # REF constraint
                    model.addConstr(new_envy_df.iloc[i, i_rm] + new_envy_df.iloc[i_rm, i] - p[i_rm] - p[i] >= new_envy_df.iloc[j, j_rm] + new_envy_df.iloc[j_rm, j] - p[j_rm] - p[j])
                    
        # LP will maximize the minimum rent
        model.addConstr(R == gp.min_(p), name="maximin")
        model.setObjective(R, GRB.MAXIMIZE)
        model.Params.LogToConsole = 0
        model.optimize()
        if model.Status >= 3:
            return []
        else:
            return [[v.X for v in model.getVars() if v.VarName != "r"], inv_epsilon]
    else: 
        # equal rent-share situation
        if share_rent_equally:
            prices = range(m)
            model = gp.Model()
            p = model.addVars(prices, vtype=GRB.CONTINUOUS, name="p")
            i_epsilon = model.addVar(vtype=GRB.CONTINUOUS, name="e")
            model.addConstr(p.sum() == 1)
            model.addConstr(i_epsilon >= 0) # envy must be non-negative
            new_envy_df = pd.DataFrame(normalize(envy_df, axis=1, norm='l1'))
            for i in range(m):
                for j in range(m):
                    i_rm = int(person_rm_table[i])
                    j_rm = int(person_rm_table[j])
                    if j != i_rm and i != j_rm:
                        # epsilon-EF constraint
                        model.addConstr(envy_df.iloc[i, i_rm] - p[i] >= i_epsilon * (envy_df.iloc[i, j] - p[j]))
                        model.addConstr(p[i] == p[i_rm]) # require that the roommates pay the same price
                        # REF constraint
                        model.addConstr(new_envy_df.iloc[i, i_rm] + new_envy_df.iloc[i_rm, i] - p[i_rm] - p[i] >= new_envy_df.iloc[j, j_rm] + new_envy_df.iloc[j_rm, j] - p[j_rm] - p[j])
                        
            # LP will maximize inverse epsilon, which will minimize epsilon (max envy bound)
            model.setObjective(i_epsilon, GRB.MAXIMIZE)
            model.Params.LogToConsole = 0
            model.optimize()
            if model.Status >= 3:
                return []
            else:
                return [[v.X for v in model.getVars() if v.VarName != "e"], [v.X for v in model.getVars() if v.VarName == "e"]]
        else:
            prices = range(m)
            model = gp.Model()
            p = model.addVars(prices, vtype=GRB.CONTINUOUS, name="p")
            i_epsilon = model.addVar(vtype=GRB.CONTINUOUS, name="e")
            model.addConstr(p.sum() == 1)
            model.addConstr(i_epsilon >= 0)
            new_envy_df = pd.DataFrame(normalize(envy_df, axis=1, norm='l1'))
            for i in range(m):
                for j in range(m):
                    i_rm = int(person_rm_table[i])
                    j_rm = int(person_rm_table[j])
                    if j != i_rm and i != j_rm:
                        # epsilon-EF constraint
                        model.addConstr(envy_df.iloc[i, i_rm] - p[i] >= i_epsilon * (envy_df.iloc[i, j] - p[j]))
                        # REF constraint
                        model.addConstr(new_envy_df.iloc[i, i_rm] + new_envy_df.iloc[i_rm, i] - p[i_rm] - p[i] >= new_envy_df.iloc[j, j_rm] + new_envy_df.iloc[j_rm, j] - p[j_rm] - p[j])
            model.setObjective(i_epsilon, GRB.MAXIMIZE)
            model.Params.LogToConsole = 0
            model.optimize()
            if model.Status >= 3:
                return []
            else:
                return [[v.X for v in model.getVars() if v.VarName != "e"], [v.X for v in model.getVars() if v.VarName == "e"] ]
            

# Runs num_simulations simulations of generating random preference matrices and finding the greedy/MWIS solutions and their prices
def simulate_ef_prices(m, n, num_simulations, inv_epsilon=None, share_rent_equally=False):
    greedy_match_results = []
    greedy_envies = []
    mwis_results = []
    mwis_envies = []

    # tabulate the number of found solutions for greedy/MWIS
    greedy_match_found_sol = 0
    mwis_found_sol = 0
    for _ in range(num_simulations):
        # generate random preference matrices with alpha = 0.5 (people generally prefer singles)
        prefs = generate_preferences(m, n, 0.5)
        greedy_utilities_df = generate_greedy_utilities(prefs)

        # create the greedy with bipartite matching partitions
        greedy_assignment_partition, greedy_assignment_result = create_greedy_partition(m, n, greedy_utilities_df)
        greedy_match_partition, greedy_match_result = greedy_match(greedy_assignment_partition, greedy_utilities_df)

        # create the MWIS partition
        mwis_partition, mwis_result = generate_mwis_solution(m, n, greedy_utilities_df)

        # generate the envy dataframe and prices from the greedy partition
        greedy_person_rm_table, greedy_person_room_table, greedy_envy_df = get_ef_df(m, greedy_match_partition, prefs)
        greedy_prices = find_ef_price(m, greedy_person_rm_table, greedy_envy_df, inv_epsilon, share_rent_equally)

        if len(greedy_prices) > 0:
            greedy_match_found_sol += 1
            greedy_match_results.append(greedy_prices)
            greedy_envies.append(find_envy_distribution(m, greedy_person_rm_table, greedy_envy_df, greedy_prices[0]))

        # generate the envy dataframe and prices from the MWIS partition
        mwis_person_rm_table, mwis_person_room_table, mwis_envy_df = get_ef_df(m, mwis_partition, prefs)
        mwis_prices = find_ef_price(m, mwis_person_rm_table, mwis_envy_df, inv_epsilon, share_rent_equally)
        
        if len(mwis_prices) > 0:
            mwis_found_sol += 1
            mwis_results.append(mwis_prices)
            mwis_envies.append(find_envy_distribution(m, mwis_person_rm_table, mwis_envy_df, mwis_prices[0]))

    return greedy_match_results, mwis_results, greedy_envies, mwis_envies, greedy_match_found_sol, mwis_found_sol

# Finds the final envy distribution of all tenants given their prices
def find_envy_distribution(m, person_rm_table, envy_df, prices):
    envies = []
    # iterate throught all tenant-tenant pairs and find their empirical envy
    for i in range(m):
        for j in range(m):
            i_rm = int(person_rm_table[i])
            j_rm = int(person_rm_table[j])
            if i != j and j != i_rm and i != j_rm:
                p_i, p_j = prices[i], prices[j]
                envy = (envy_df.iloc[i, j] - p_j)/(envy_df.iloc[i, i_rm] - p_i)
                envies.append(envy)
    return envies

