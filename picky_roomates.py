import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from simulate import *
from mwis import *
from greedy import *
from ef import *

"""## Greedy and Greedy + Matching"""
num_trials = 10
num_simulations = 200
sim_df = pd.DataFrame(columns=["m", "n", "trial", "greedy_ratio", "greedy_match_ratio"])
for i in trange(num_trials):
    for n in range(2, 5):
        for m in range(n, 2 * n + 1):
            greedy_scores, greedy_match_scores, optimal_scores = simulate(m, n, num_simulations, 0)
            sim_df.loc[len(sim_df)] = {'m': m, 'n': n, 'trial': i, 'greedy_ratio': np.mean(np.divide(greedy_scores, optimal_scores)), 'greedy_match_ratio': np.mean(np.divide(greedy_match_scores, optimal_scores))}

m, n = 4, 3
num_trials = 10
num_simulations = 200
sim_df_alpha = pd.DataFrame(columns=["alpha", "trial", "greedy_ratio", "greedy_match_ratio"])
for i in trange(num_trials):
    for alpha in np.arange(0, 1.1, 0.1):
        greedy_scores, greedy_match_scores, optimal_scores = simulate(m, n, num_simulations, alpha)
        sim_df_alpha.loc[len(sim_df_alpha)] = {'alpha': alpha, 'trial': i, 'greedy_ratio': np.mean(np.divide(greedy_scores, optimal_scores)), 'greedy_match_ratio': np.mean(np.divide(greedy_match_scores, optimal_scores))}

sim_df.to_csv("roommate_sim_greedy.csv")
sim_df_alpha.to_csv("roommate_sim_greedy_with_alpha.csv")

sort_idx = np.argsort(np.divide(greedy_scores, optimal_scores))
plt.plot(np.divide(greedy_scores, optimal_scores)[sort_idx])
plt.plot(np.divide(greedy_match_scores, optimal_scores)[sort_idx])

plt.hist(np.divide(greedy_scores, optimal_scores), bins=50)
plt.hist(np.divide(greedy_match_scores, optimal_scores), bins=50)

"""## MWIS with m = 2n"""
num_trials = 10

sim_df_mwis = pd.DataFrame(columns=["n", "trial", "greedy", "greedy_match", "mwis", "optimal"])
for n in range(2, 8):
    for i in trange(num_trials):
        m = 2 * n
        if n <= 5:
            greedy_scores, greedy_match_scores, mwis_scores, optimal_scores = simulate_mwis(m, n, num_simulations)
        else:
            greedy_scores, greedy_match_scores, mwis_scores, optimal_scores = simulate_mwis(m, n, num_simulations, do_optimal=False)
            optimal_scores = 0

        sim_df_mwis.loc[len(sim_df_mwis)] = {'n': n, 'trial': i, 'greedy': np.mean(greedy_scores), 'greedy_match': np.mean(greedy_match_scores), 'mwis': np.mean(mwis_scores), 'optimal': np.mean(optimal_scores)}

sim_df_mwis.to_csv('roommate_sim_mwis.csv')

m, n, num_simulations = 6, 3, 1000
greedy_scores, greedy_match_scores, mwis_scores, optimal_scores = simulate_mwis(m, n, num_simulations)

sort_idx = np.argsort(np.divide(greedy_match_scores, optimal_scores))
#plt.plot(np.divide(greedy_scores, optimal_scores)[sort_idx])
plt.plot(np.divide(greedy_match_scores, optimal_scores)[sort_idx])
plt.plot(np.divide(mwis_scores, optimal_scores)[sort_idx])
plt.legend(["Greedy with matching", "MWIS"])

match_divide = np.divide(greedy_match_scores, optimal_scores)[sort_idx]
mwis_divide = np.divide(mwis_scores, optimal_scores)[sort_idx]
pd.concat([pd.DataFrame(match_divide), pd.DataFrame(mwis_divide)]).to_csv('mwis_indiv.csv')

# MWIS runtime simulation
timings = runtime_sim()
z = np.polyfit(x=np.array(range(2, 12)), y=np.array(timings), deg=3)
p3 = np.poly1d(z)

x = np.linspace(2, 11, 10)
plt.plot(x, timings)
plt.plot(x, p3(x))
plt.legend(["sim timing", "polyfit O(n^3)"])

pd.DataFrame(timings).to_csv("runtime.csv")

"""## MWIS with m < 2n"""
num_trials = 3
num_simulations = 100
#sim_scores = {}
sim_df_ghosts = pd.DataFrame(columns=["m", "n", "trial", "greedy_ratio", "greedy_match_ratio", "mwis_ghost_ratio"])
for i in trange(num_trials):
    for n in range(2, 5):
        for m in trange(n, 2 * n + 1):
            greedy_scores, greedy_match_scores, mwis_scores, optimal_scores = simulate_mwis_with_ghosts(m, n, num_simulations, 0.5)

            sim_df_ghosts.loc[len(sim_df_ghosts)] = {'m': m, 'n': n, 'trial': i, 'greedy_ratio': np.mean(np.divide(greedy_scores, optimal_scores)), 'greedy_match_ratio': np.mean(np.divide(greedy_match_scores, optimal_scores)), 'mwis_ghost_ratio': np.mean(np.divide(mwis_scores, optimal_scores))}

sim_df_ghosts.to_csv("sim_df_ghosts.csv")

"""## Envy-Freeness"""
gm_num_found = []
gm_price_min = []
mwis_num_found = []
mwis_price_min = []

num_simulations = 200
for alpha in np.arange(0.1, 1.1, 0.1):
    greedy_match_results, mwis_results, greedy_match_found_sol, mwis_found_sol = simulate_ef_prices(6, 3, num_simulations, alpha)
    gm_num_found.append(greedy_match_found_sol)
    mwis_num_found.append(mwis_found_sol)
    gm_price_min.append([min(x) for x in greedy_match_results])
    mwis_price_min.append([min(x) for x in mwis_results])

plt.plot(np.arange(0.1, 1.1, 0.1), gm_num_found)
plt.plot(np.arange(0.1, 1.1, 0.1), mwis_num_found)
plt.legend(["Greedy+Matching", "MWIS"])
plt.xlabel("alpha-EF")
plt.ylabel("%Found EFX Solution")

plt.hist(gm_price_min[4])
plt.hist(mwis_price_min[4])

pd.DataFrame(gm_num_found).to_csv("gm_num_found_M6.csv")
pd.DataFrame(mwis_num_found).to_csv("mwis_num_found_M6.csv")
pd.DataFrame(gm_price_min).to_csv("gm_price_min_M6.csv")
pd.DataFrame(mwis_price_min).to_csv("mwis_price_min_M6.csv")