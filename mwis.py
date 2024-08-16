import networkx as nx
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import itertools
import json

f = open("license.lic")
env = gp.Env(params=json.load(f))

# Generates the maximum weighted independent set (MWIS) solution from an adjacency matrix
# and corresponding weights
def maximum_weighted_independent_set(adjacency_matrix, weights, env):
    with gp.Model("mwis", env=env) as model:
        rows, cols = adjacency_matrix.tocoo().row, adjacency_matrix.tocoo().col
        num_vertices, num_edges = len(weights), len(rows)
        
        # create Gurobi model and add each vertex of the graph as a variable
        x = model.addMVar(num_vertices, vtype=GRB.BINARY, name="x")
        # maximize the sum of the vertex weights in the independent set
        model.setObjective(weights @ x, sense=GRB.MAXIMIZE)

        # generate a sparse array to run IP solver
        indices = []
        for i, j in zip(rows, cols):
            indices.extend([i, j])
        indptr = range(0, len(indices) + 2, 2)
        data = np.ones(2 * num_edges)
        A = sp.csc_array((data, indices, indptr), shape=(num_vertices, num_edges))

        # IP Solver for MWIS
        model.addMConstr(
            A.T,
            x,
            GRB.LESS_EQUAL,
            np.ones(A.shape[1]),
            name="no_adjacent_vertices",
        )
        model.Params.LogToConsole = 0
        model.optimize()
        (mwis,) = np.where(x.X >= 0.5)
        return mwis, sum(weights[mwis])

# Generate all edges of the graph for a roommate problem with m tenants and n rooms
def generate_edges(m, n):
    edges = []
    
    # iterate over all possible roommate/room combinations and create corresponding edges
    for i in range(1, m + 1):
        for j in range(i + 1, m + 1):
            for r in range(n):
                edges.append((i, j, r))
                
    # trim duplicate edges (same rooming group, etc.)
    possible_edges_before_trimming = list(itertools.combinations(edges, 2))
    trimmed_edges = [x for x in possible_edges_before_trimming if (x[0][0] == x[1][0]) or (x[0][1] == x[1][1]) or (x[0][2] == x[1][2]) or (x[0][0] == x[1][1]) or (x[0][1] == x[1][0])]
    return trimmed_edges

# Generate the MWIS solution using Gurobi solver from a greedy utilities matrix
def generate_mwis_solution(m, n, greedy_df):
    G = nx.Graph()
    nodes = [((x[0], x[1], x[2]), {"utility":x[3]}) if (x[0] < x[1]) else ((x[1], x[0], x[2]), {"utility":x[3]}) for x in np.array(greedy_df[["person1", "person2", "room", "utility"]])]
    G.add_nodes_from(nodes)
    G.add_edges_from(generate_edges(m, n))
    G.remove_nodes_from(list(nx.isolates(G)))

    # add utility values as the edge weights
    weights = np.array(list(nx.get_node_attributes(G, "utility").values()))
    mwis, mwis_result = maximum_weighted_independent_set(nx.adjacency_matrix(G), weights, env)
    
    # create the MWIS partition
    mwis_partition = np.array(G.nodes())[mwis]
    return mwis_partition, mwis_result

# Generate the preference matrix with "ghosts" (m < 2n)
def add_preferences_with_ghosts(m, n, prefs):
    ghost_preference_matrix = []

    if m == 2 * n:
        return prefs

    num_ghosts = 2 * n - m
    
    # iterate over original tenants and half their single utilities for rooming with a ghost
    for i in range(m):
        player_preference = np.zeros((2 * n, n))
        player_preference[:m] = prefs[i]

        for j in range(num_ghosts):
            player_preference[m + j] = player_preference[i] / 2

        ghost_preference_matrix.append(player_preference)
        
    # iterate over the ghost tenants which have half the utility of rooming with a non-ghost tenant
    for i in range(m, 2 * n):
        player_preference = np.zeros((2 * n, n))
        for j in range(m):
            player_preference[j] = ghost_preference_matrix[j][i]

        # ghosts should not live with other ghosts -- generate very negative utilities to prevent ghosts from being paired up
        for j in range(m, 2 * n):
            player_preference[j] = np.ones(n) * -100

        ghost_preference_matrix.append(player_preference)

    return ghost_preference_matrix