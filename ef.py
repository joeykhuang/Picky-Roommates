
from mwis import generate_mwis_solution
from greedy import create_greedy_partition, greedy_match, generate_greedy_utilities
from preferences import generate_preferences
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from sklearn.preprocessing import normalize

def get_ef_df(m, partition, prefs):
    person_room_table = {}
    person_rm_table = {}
    envy_table = []

    for part in partition:
        p1, p2, r = part
        person_room_table.update({p1: r, p2: r})
        person_rm_table.update({p1 - 1: p2 - 1, p2 - 1: p1 - 1})

    for i in range(1, m + 1):
        i_utilities = []
        for j in range(1, m + 1):
            j_room = int(person_room_table[j])
            roommate_utility = prefs[i - 1][j - 1][j_room]
            i_utilities.append(roommate_utility)
        envy_table.append(i_utilities)

    envy_df = pd.DataFrame(envy_table)
    return person_rm_table, person_room_table, envy_df

def find_ef_price(m, person_rm_table, envy_df, inv_epsilon = None, share_rent_equally=False):
    if inv_epsilon != None:
        prices = range(m)
        model = gp.Model()
        p = model.addVars(prices, vtype=GRB.CONTINUOUS, name="p")
        R = model.addVar(vtype=GRB.CONTINUOUS, name="r")
        model.addConstr(p.sum() == 1)
        new_envy_df = pd.DataFrame(normalize(envy_df, axis=1, norm='l1'))
        for i in range(m):
            for j in range(m):
                i_rm = int(person_rm_table[i])
                j_rm = int(person_rm_table[j])
                if j != i_rm and i != j_rm:
                    # PEF
                    model.addConstr(envy_df.iloc[i, i_rm] - p[i] >= inv_epsilon * (envy_df.iloc[i, j] - p[j]))
                    # REF
                    model.addConstr(new_envy_df.iloc[i, i_rm] + new_envy_df.iloc[i_rm, i] - p[i_rm] - p[i] >= new_envy_df.iloc[j, j_rm] + new_envy_df.iloc[j_rm, j] - p[j_rm] - p[j])
        model.addConstr(R == gp.min_(p), name="maximin")
        model.setObjective(R, GRB.MAXIMIZE)
        model.Params.LogToConsole = 0
        model.optimize()
        if model.Status >= 3:
            return []
        else:
            return [[v.X for v in model.getVars() if v.VarName != "r"], inv_epsilon]
    else: 
        if share_rent_equally:
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
                        # PEF
                        model.addConstr(envy_df.iloc[i, i_rm] - p[i] >= i_epsilon * (envy_df.iloc[i, j] - p[j]))
                        model.addConstr(p[i] == p[i_rm])
                        # REF
                        model.addConstr(new_envy_df.iloc[i, i_rm] + new_envy_df.iloc[i_rm, i] - p[i_rm] - p[i] >= new_envy_df.iloc[j, j_rm] + new_envy_df.iloc[j_rm, j] - p[j_rm] - p[j])
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
                        # PEF
                        model.addConstr(envy_df.iloc[i, i_rm] - p[i] >= i_epsilon * (envy_df.iloc[i, j] - p[j]))
                        # REF
                        model.addConstr(new_envy_df.iloc[i, i_rm] + new_envy_df.iloc[i_rm, i] - p[i_rm] - p[i] >= new_envy_df.iloc[j, j_rm] + new_envy_df.iloc[j_rm, j] - p[j_rm] - p[j])
            model.setObjective(i_epsilon, GRB.MAXIMIZE)
            model.Params.LogToConsole = 0
            model.optimize()
            if model.Status >= 3:
                return []
            else:
                return [[v.X for v in model.getVars() if v.VarName != "e"], [v.X for v in model.getVars() if v.VarName == "e"] ]
            


def simulate_ef_prices(m, n, num_simulations, inv_epsilon=None, share_rent_equally=False):
    greedy_match_results = []
    greedy_envies = []
    mwis_results = []
    mwis_envies = []

    greedy_match_found_sol = 0
    mwis_found_sol = 0
    for _ in range(num_simulations):
        prefs = generate_preferences(m, n, 0.5)
        greedy_utilities_df = generate_greedy_utilities(prefs)

        greedy_assignment_partition, greedy_assignment_result = create_greedy_partition(m, n, greedy_utilities_df)
        greedy_match_partition, greedy_match_result = greedy_match(greedy_assignment_partition, greedy_utilities_df)

        mwis_partition, mwis_result = generate_mwis_solution(m, n, greedy_utilities_df)

        greedy_person_rm_table, greedy_person_room_table, greedy_envy_df = get_ef_df(m, greedy_match_partition, prefs)
        greedy_prices = find_ef_price(m, greedy_person_rm_table, greedy_envy_df, inv_epsilon, share_rent_equally)

        if len(greedy_prices) > 0:
            greedy_match_found_sol += 1
            greedy_match_results.append(greedy_prices)
            greedy_envies.append(find_envy_distribution(m, greedy_person_rm_table, greedy_person_room_table, greedy_envy_df, greedy_prices[0]))

        mwis_person_rm_table, mwis_person_room_table, mwis_envy_df = get_ef_df(m, mwis_partition, prefs)
        mwis_prices = find_ef_price(m, mwis_person_rm_table, mwis_envy_df, inv_epsilon, share_rent_equally)
        

        if len(mwis_prices) > 0:
            mwis_found_sol += 1
            mwis_results.append(mwis_prices)
            mwis_envies.append(find_envy_distribution(m, mwis_person_rm_table, mwis_person_room_table, mwis_envy_df, mwis_prices[0]))

    return greedy_match_results, mwis_results, greedy_envies, mwis_envies, greedy_match_found_sol, mwis_found_sol

def find_envy_distribution(m, person_rm_table, person_room_table, envy_df, prices):
    envies = []
    for i in range(m):
        for j in range(m):
            i_rm = int(person_rm_table[i])
            j_rm = int(person_rm_table[j])
            if i != j and j != i_rm and i != j_rm:
                p_i, p_j = prices[i], prices[j]
                envy = (envy_df.iloc[i, j] - p_j)/(envy_df.iloc[i, i_rm] - p_i)
                envies.append(envy)
    return envies

