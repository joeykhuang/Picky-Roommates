
from mwis import generate_mwis_solution
from greedy import create_greedy_partition, greedy_match, generate_greedy_utilities
from preferences import generate_preferences
import numpy as np
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
            if i != j:
                j_room = int(person_room_table[j])
                roommate_utility = prefs[i - 1][j - 1][j_room]
                i_utilities.append(roommate_utility)
            else:
                i_utilities.append(0)
        envy_table.append(i_utilities)

    envy_df = pd.DataFrame(envy_table)
    return person_rm_table, envy_df

def find_ef_price(m, partition, person_rm_table, envy_df, alpha):
    prices = range(m)
    model = gp.Model()
    p = model.addVars(prices, vtype=GRB.CONTINUOUS, name="p")
    R = model.addVar(vtype=GRB.CONTINUOUS, name="r")
    model.addConstr(p.sum() == 1)
    new_envy_df = pd.DataFrame(normalize(envy_df, axis=1, norm='l1'))
    for i in range(m):
        for j in range(i + 1, m):
            i_rm = int(person_rm_table[i])
            j_rm = int(person_rm_table[j])
            if j != i_rm and i != j_rm:
                # PEF
                model.addConstr(envy_df.iloc[i, i_rm] - p[i] >= alpha * (envy_df.iloc[i, j_rm] - p[j]))
                # REF
                model.addConstr(new_envy_df.iloc[i, i_rm] + new_envy_df.iloc[i_rm, i] - p[i_rm] - p[i] >= new_envy_df.iloc[j, j_rm] + new_envy_df.iloc[j_rm, j] - p[j_rm] - p[j])
    model.addConstr(R == gp.min_(p), name="minimax")
    model.setObjective(R, GRB.MAXIMIZE)
    model.Params.LogToConsole = 0
    model.optimize()
    if model.Status >= 3:
        return []
    else:
        return [v.X for v in model.getVars() if v.VarName != "r"]


def simulate_ef_prices(m, n, num_simulations, alpha):
    greedy_match_results = []
    mwis_results = []

    greedy_match_found_sol = 0
    mwis_found_sol = 0
    for _ in range(num_simulations):
        prefs = generate_preferences(m, n, 0.5)
        greedy_utilities_df = generate_greedy_utilities(prefs)

        greedy_assignment_partition, greedy_assignment_result = create_greedy_partition(m, n, greedy_utilities_df)
        greedy_match_partition, greedy_match_result = greedy_match(greedy_assignment_partition, greedy_utilities_df)

        mwis_partition, mwis_result = generate_mwis_solution(m, n, greedy_utilities_df)

        greedy_person_rm_table, greedy_envy_df = get_ef_df(m, greedy_match_partition, prefs)
        greedy_prices = find_ef_price(m, greedy_match_partition, greedy_person_rm_table, greedy_envy_df, alpha)

        if len(greedy_prices) > 0:
            greedy_match_found_sol += 1
            greedy_match_results.append(greedy_prices)

        mwis_person_rm_table, mwis_envy_df = get_ef_df(m, mwis_partition, prefs)
        mwis_prices = find_ef_price(m, mwis_partition, mwis_person_rm_table, mwis_envy_df, alpha)

        if len(mwis_prices) > 0:
            mwis_found_sol += 1
            mwis_results.append(mwis_prices)

    return greedy_match_results, mwis_results, greedy_match_found_sol, mwis_found_sol