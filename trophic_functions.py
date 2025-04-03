# -*- coding: utf-8 -*-
"""Trophic coherence

Lucy Dhumeaux
03.04.2025

There is two plot functions: 
    -one for networks with positive weights only
    -one for networks with negative weights
There are also two of each, for basal and no-basal networks: 
    -calculate incoherence
    -calculate trophic levels
    -generate a preferential preying network
    -simulate majority rule evolution
There are two classes for training:
    -the usual Hebbian rule
    -the iterative Hebbian rule
And a function to tune the incoherence.
"""

#All imports
import numpy as np
import networkx as nx
import random
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from itertools import combinations
import threading
import time


"""Main functions for basal networks"""

def plot_trophic_level(G, trophic_levels,
                       k=1,        # Spring constant; larger => more spread out
                       iterations=20,# Increase iterations for refinement
                       xscale=5.0    # Horizontal scaling factor
                      ):
    """
    Plots graph G where y-positions are set by trophic_levels[node].
    The x-positions are derived from a spring layout but then rescaled
    to spread nodes horizontally.

    Args:
      G: An nx graph
      trophic_levels: dict {node: trophic_level}
      k: Parameter for nx.spring_layout controlling spacing (default 0.5)
      iterations: Iterations for spring_layout (default 50)
      xscale: Final scale factor applied to x-coordinates
    """
    pos = nx.spring_layout(G, k=k, iterations=iterations, seed=42)

    # Override the y-coordinates using trophic_levels
    for node in list(pos.keys()):
        if node not in trophic_levels:
            # Bit of a cop out here so might raise issues
            pos.pop(node)
            continue

        pos[node][1] = trophic_levels[node]  # fix the y-coordinate

        # Put isolated nodes at y = -1
        if G.out_degree(node) == 0 and G.in_degree(node) == 0:
            pos[node][1] = -1

    # Rescale the x-coordinates to reduce risk of bunching
    if pos:
        xs = [pos[n][0] for n in pos]
        min_x, max_x = min(xs), max(xs)
        if max_x > min_x:  # avoid dividing by zero in trivial graphs
            for n in pos:
                # shift to [0..1], then multiply by xscale
                pos[n][0] = (pos[n][0] - min_x) / (max_x - min_x) * xscale

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='indigo', alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='#9370DB', arrows=True)
    plt.gca().yaxis.set_ticks_position('left')
    plt.grid(False)
    plt.ylabel("Trophic Level")

    return plt




def calculate_trophic_levels(G):
    """

    Args:
      G: An nx graph

    Returns: A dictionnary with nodes as keys and their trophic level as values

    """

    N = G.number_of_nodes()
    # Create the adjacency matrix A
    node_order = sorted(G.nodes)
    A = nx.to_numpy_array(G, nodelist=node_order)

    # Calculate z vector
    in_degrees = np.sum(A, axis=0)  # sum over cols for in-degrees
    z = np.maximum(in_degrees, 1)

    Z = np.diag(z)
    Lambda = Z - A.T
    # Solve matrix equation
    try:
        Lambda_inv = np.linalg.inv(Lambda)
        s = np.dot(Lambda_inv, z)
    except np.linalg.LinAlgError:
        print("Lambda was singular; gave up")
        s = np.array([0 for _ in range(N)])

    # Writing trophic levels in dict
    trophic_levels = {n: s[n] for n in range(N)}

    return trophic_levels



def incoherence(G, trophic_level):
  """

  Args:
    trophic_level: A dictionnary with nodes as keys and their trophic level as values

  Returns: The incoherence of the trophic levels, calculated as the standard deviation
  of the trophic differences between each node

  """

  # Calculate the trophic difference matrix x_ij = s_j - s_i
  s = np.array(list(trophic_level.values()))
  x_ij = s[None, :] - s[:, None]

  # Calculate incoherence parameter
  edges = list(G.edges())
  trophic_differences = np.array([x_ij[i, j] for i, j in edges]) # taking only difference of existing edges
  q = np.std(trophic_differences)

  if q < 1e-8:
    q = 0
  return q

"""Example"""
# Adjacency matrix as a NumPy array
adj_matrix = np.array([[0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])

# Convert the NumPy array to a NetworkX graph
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

trophic_level = calculate_trophic_levels(G)
q = incoherence(G, trophic_level)
print(q)
plt = plot_trophic_level(G, trophic_level)
plt.show()



"""Main functions for no-basal networks"""

def nobasal_calculate_trophic_levels(A):
    """

    Args:
      A: adjencey matrix of the graph, should already be transposed

    Returns: A dictionnary with nodes as keys and their trophic level as values

    """
    N = A.shape[0]
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    # Calculate z vector
    in_degrees = np.sum(A, axis=0)  # sum over columns(!!) for in-degrees
    out_degrees = np.sum(A, axis=1)
    z = in_degrees + out_degrees
    v = (in_degrees - out_degrees)

    Z = np.diag(z)
    Lambda = Z - A - np.transpose(A)

    # Flag to stop the monitoring thread
    stop_monitor = threading.Event()

    def monitor_execution(delay, message):
        if not stop_monitor.wait(delay):  # Wait for delay or an event to stop
            print(message)

    # Start a monitoring thread
    timeout_seconds = 5
    monitor_thread = threading.Thread(target=monitor_execution, args=(timeout_seconds, f"Computation (solving TL eq) is taking longer than {timeout_seconds} seconds..."))
    monitor_thread.start()

    # Solve matrix equation
    try:
        Lambda_inv = np.linalg.pinv(Lambda)
        s = np.dot(Lambda_inv, v)
    except np.linalg.LinAlgError:
        print("Lambda was singular; tried a different method.")
        epsilon = 1e-6
        Lambda = Z - A + epsilon * np.eye(N)
        Lambda_inv = np.linalg.inv(Lambda)
        s = Lambda_inv @ z

    # Signal the monitoring thread to stop and wait for it to finish
    stop_monitor.set()
    monitor_thread.join()

    # Shifting so that the lowest level is 0
    shift = np.min(s)
    s = s - shift

    # For nodes just self-connected, set trophic level to 0
    for n in range(N):
      if G.has_edge(n, n) and G.degree(n) == 1:
        print(n,'self connected')
        s[n] = 0

    # Writing trophic levels in dict
    trophic_levels = {n: s[n] for n in range(N)}

    return trophic_levels



def nobasal_incoherence(A, trophic_level):
    N = A.shape[0]

    # Ignore self-connections
    np.fill_diagonal(A, 0)

    # Convert trophic levels to a NumPy array in the same order
    s = np.around(np.array([trophic_level[node] for node in range(N)]), decimals=14)

    # Vectorized computation of |s[j] - s[i]| - 1
    # Broadcasting s as a column vector and row vector
    x_ij = s[None, :] - s[:, None]
    diff = x_ij - 1  # shape: (N, N)

    numerator = np.sum(A * (diff**2))
    denominator = np.sum(A)
    F = numerator / denominator if denominator != 0 else 0

    if F < 1e-8:
      F = 0

    return F


"""Example"""
# Adjacency matrix as a NumPy array
A = np.array([[0, 0, 0, 1],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [1, 1, 0, 0]])
# Convert the NumPy array to a NetworkX graph
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
trophic_level_nobasal = nobasal_calculate_trophic_levels(A)
F_nobasal = nobasal_incoherence(A, trophic_level_nobasal)
print('Incoherence parameter F=', F_nobasal)

# Plot the network with trophic coherence
plt = plot_trophic_level(G, trophic_level_nobasal)
plt.show()



"""Preferential preying models"""

def preferential_preying_model(N, B, L, T):
    # Initialize the network with B basal vertices
    G = nx.DiGraph()
    basal_nodes = range(N-B,N)
    G.add_nodes_from(basal_nodes)
    # Assign initial trophic levels for basal vertices
    trophic_levels = {node: 1 for node in basal_nodes}

    # Add non-basal vertices with one random in-neighbor
    non_basal_nodes = range(N-B)
    for v in non_basal_nodes:
        # Choose a random existing vertex as the in-neighbor
        in_neighbor = random.choice(list(G.nodes))

        G.add_node(v)
        G.add_edge(in_neighbor, v)
        trophic_levels[v] = trophic_levels[in_neighbor] + 1

    # Add additional edges to reach a total of L edges
    existing_edges = len(G.edges)
    max_retries = 10**6
    retries = 0

    while existing_edges < L:
      if retries >= max_retries:
          print(f"Max retries reached ({max_retries}). Function terminated early. Number of edges: {existing_edges}")
          break

      # if existing_edges % 500 == 0: #update levels every 500 steps
      #       trophics = calculate_trophic_levels(G)
      #       trophic_levels = list(trophics.values())

      # Randomly select non-basal node pairs
      j = random.choice(non_basal_nodes)
      i = random.choice(range(N))
      if not G.has_edge(i, j) and i != j:  # Avoid duplicate edges and self connections
          x_ij = trophic_levels[j] - trophic_levels[i]
          prob = np.exp(-((x_ij - 1) ** 2) / (2 * T ** 2))
          # With probability `prob`, add the directed edge
          if random.random() < prob:
              G.add_edge(i, j)
              existing_edges += 1
          else: retries += 1
      else: retries += 1

    return G

def nobasal_preferential_preying_model(N, L, T, trophic_function):
    G = nx.DiGraph()
    nodes = range(N)
    G.add_nodes_from(range(N))
    # Add non-basal vertices with one random in-neighbor
    for node in nodes:
        # Choose a random existing vertex as the in-neighbor
        in_neighbor = random.choice(list(G.nodes))
        G.add_edge(in_neighbor, node)

    A = nx.to_numpy_array(G, nodelist=nodes)
    trophics = trophic_function(A)
    trophic_levels = list(trophics.values())

    # Add additional edges to reach a total of L edges
    existing_edges = len(G.edges)
    max_retries = 10**6
    retries = 0

    while existing_edges < L:
      if retries >= max_retries:
          print(f"Max retries reached ({max_retries}). Function terminated early. Number of edges: {existing_edges}")
          break

      # Randomly select node pairs
      i, j = random.sample(nodes, 2)
      if not G.has_edge(i, j):  # Avoid duplicate edges
          x_ij = trophic_levels[j] - trophic_levels[i]

          prob = np.exp(-((x_ij - 1) ** 2) / (2 * T ** 2))
          # With probability `prob`, add the directed edge
          if random.random() < prob:
              G.add_edge(i, j)
              existing_edges += 1
              if existing_edges % 500 == 0: #update levels every 500 steps
                A = nx.to_numpy_array(G, nodelist=nodes)
                trophics = trophic_function(A)
                trophic_levels = list(trophics.values())

          else: retries += 1
      else: retries += 1

    # Checking for basals
    for node in nodes:
      print(node,"is basal") if G.in_degree(node) == 0 else None

    return G



"""Examples"""
# Parameters
N = 100
B = 5
L = 500
T = 1  # Temperature parameter

G_pp = preferential_preying_model(N, B, L, T)
trophic_level_pp = calculate_trophic_levels(G_pp)
q_pp = incoherence(G_pp, trophic_level_pp)
print('Incoherence parameter q=',q_pp)

# Plot the network with trophic coherence
plot_trophic_level(G_pp, trophic_level_pp)
plt.show()


# Parameters
N = 100  # Total nodes
L = 2000  # Total edges
T = 0.1  # Temperature parameter

G_pp = nobasal_preferential_preying_model(N, L, T, nobasal_calculate_trophic_levels)
A = nx.to_numpy_array(G_pp, nodelist=range(N))
trophic_level_pp = nobasal_calculate_trophic_levels(A)
F = nobasal_incoherence(A, trophic_level_pp)
print('Incoherence F=', F)

# Plot the network with trophic coherence
plot_trophic_level(G_pp, trophic_level_pp)
plt.show()




"""Majority rule dynamics"""

def mean_activity(G, steps):
  """
  A variable sigma = +- 1 is assigned to each node. At each time step sigma is updated
  according to the majority rule. The mean activity is the mean of all sigmas.
  Args:
    G: An nx graph
    time: Running time

  Returns: A list of the mean activity at each time t

  """
  node_order = sorted(G.nodes)
  A = nx.to_numpy_array(G, nodelist=node_order)
  N = G.number_of_nodes()

  sig = np.random.choice([-1, 1], size=N).reshape(-1,1)
  m = []

  for t in range(steps):
    m.append(np.mean(sig))
    h = A.T @ sig
    for i in range(N):
      sig[i] = 1 if h[i] > 0 else (-1 if h[i] < 0 else np.random.choice([-1, 1]))

  return m


def m_flips(mean_activity, steps):
    m = np.array(mean_activity)
    signs = np.sign(m)
    sign_changes = np.diff(signs)
    sign_flips = np.count_nonzero(sign_changes)

    p_flip = sign_flips / steps

    return sign_flips


def nobasal_mean_activity(A, step, TL, s_tilde, theta):
  N = A.shape[0]
  node_order = range(N)

  # Determine S, the number of sensory nodes
  S = int(round(s_tilde * N))
  S = max(S, 1)  # Ensure at least one node is selected

  # Select sensory nodes: nodes with the lowest trophic levels.
  # (We use the default value 0 if a node's trophic level is not in the dictionary.)
  sensory_nodes = sorted(node_order, key=lambda node: TL.get(node, 0))[:S]

  sig = np.random.choice([-1, 1], size=N).reshape(-1,1)
  m = []

  for t in range(step):
    m.append(np.mean(sig))
    h = A.T @ sig
    count = 0
    for i in range(N):
      node = node_order[i]
      if node in sensory_nodes:
        h[i] = 0
        sig[i] = np.random.choice([-1, 1])
        count += 1
      else:
        proba = 1/2 + np.tanh(h[i]/theta)/2
        r = random.random()
        sig[i] = 1 if r < proba else -1

  return m


"""Example"""
# Parameters
N = 100  # Total nodes
B = 5 # Basal nodes
L =  1500 # Total edges
time = 1000

G0 = preferential_preying_model(N, B, L, T=0.5)
m_zero_T = mean_activity(G0, time)

G1 = preferential_preying_model(N, B, L, T=0.6)
m_one_T = mean_activity(G1, time)

G10 = preferential_preying_model(N, B, L, T=0.7)
m_ten_T = mean_activity(G10, time)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(m_zero_T)
ax1.set_title("T=0.5")
ax2.plot(m_one_T)
ax2.set_title("T=0.6")
ax3.plot(m_ten_T)
ax3.set_title("T=0.7")

plt.tight_layout()
plt.show()


# Parameters
N = 500  # Total nodes
L =  8500 # Total edges
time = 500

T0=0.1
G0 = nobasal_preferential_preying_model(N, L, T0, nobasal_calculate_trophic_levels)
A0 = nx.to_numpy_array(G0)
levels = nobasal_calculate_trophic_levels(A0)
m = max(list(levels.values()))
incoherence0 = nobasal_incoherence(A0, levels)
m_zero_T = nobasal_mean_activity(A0, time, levels, 0.1, 0.01)
print('The incoherence for T=', T0, 'is F=', incoherence0)

T1=0.75
G1 = nobasal_preferential_preying_model(N, L, T1, nobasal_calculate_trophic_levels)
A1 = nx.to_numpy_array(G1)
levels = nobasal_calculate_trophic_levels(A1)
m = max(list(levels.values()))
incoherence1 = nobasal_incoherence(A1, levels)
m_one_T = nobasal_mean_activity(A1, time, levels, 0.1, 0.01)
print('The incoherence for T=', T, 'is F=', incoherence1)

T10=2
G10 = nobasal_preferential_preying_model(N, L, T10, nobasal_calculate_trophic_levels)
A10 = nx.to_numpy_array(G10)
levels = nobasal_calculate_trophic_levels(A10)
m = max(list(levels.values()))
incoherence10 = nobasal_incoherence(A10, levels)
m_ten_T = nobasal_mean_activity(A10, time, levels, 0.1, 0.01)
print('The incoherence for T=', T, 'is F=', incoherence10)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(m_zero_T)
ax1.set_title(f"T = {T0},  F = {np.round(incoherence0, 3)}")
ax2.plot(m_one_T)
ax2.set_title(f"T = {T1},  F = {np.round(incoherence1, 3)}")
ax3.plot(m_ten_T)
ax3.set_title(f"T = {T10},  F = {np.round(incoherence10, 3)}")

plt.tight_layout()
plt.show()



"""Changing the incoherence"""

def build_Lambda_and_v(A):
    """
    Builds Lambda = Z - A - A^T, and v = in_deg - out_deg.
    Z is diagonal with z_i = (in_deg + out_deg).
    """
    in_deg = A.sum(axis=0)
    out_deg = A.sum(axis=1)
    z = in_deg + out_deg
    N = A.shape[0]

    Z = np.diag(z)
    Lambda = Z - A - A.T
    v = in_deg - out_deg
    return Lambda, v


def gradient_incoherence_wrt_h(A, h):
    """
    Returns the gradient dF/dh, where
      F = (1 / sum_{i,j} A[i,j]) * sum_{i,j} A[i,j] * (|h[i] - h[j]| - 1)^2.

    Parameters
    ----------
    A : (N,N) np.ndarray
        The adjacency matrix (can be directed or undirected).
    h : (N,)  np.ndarray
        The trophic-level vector.

    Returns
    -------
    grad : (N,) np.ndarray
        The partial derivatives of F wrt each h[k].
    """
    N = A.shape[0]
    # Denominator = sum of all adjacency weights
    denominator = A.sum()
    if denominator == 0:
        # Edge case: if A is all zeros, gradient is 0
        return np.zeros_like(h)

    # Build X = h_j - h_i
    X = h[None, :] - h[:, None]

    # Compute diff = (|X| - 1)
    diff = X - 1.0

    # Build M = A * 2 * diff * sign(X)
    #    We multiply elementwise: A * 2*(|X|-1)*sign(X)
    signX = np.sign(X)  # subgradient sign(0) = 0
    M = A * (2.0 * diff)

    # Compute row sums and column sums
    row_sums = M.sum(axis=1)  # shape (N,)
    col_sums = M.sum(axis=0)  # shape (N,)

    # gradient wrt h_k = (row_sums[k] - col_sums[k]) / denominator
    grad = -(row_sums - col_sums) / denominator

    return grad

def gradient_wrt_A(A, trophic_levels, F):
    """
    Returns (dF_dA, F_value)

    dF_dA is an NxN array whose (i,j) entry is the partial derivative
    dF / dA_{ij}.

    Steps:
      1) Compute Lambda = Z - A - A^T, and h = Lambda^{-1} v.
      2) Compute gradient of F wrt h, i.e. g_h = dF/dh.
      3) For each (i,j), compute d h / d A_{ij} using
            dLambda/dA_{ij} = - (E_{ij} + E_{ji}),
         then
            d h / d A_{ij} = - Lambda^{-1} ( dLambda/dA_{ij} ) Lambda^{-1} v
                            = + Lambda^{-1} ( E_{ij}+E_{ji} ) h
         (the two minus signs cancel).
      4) Then chain rule:  dF/dA_{ij} = g_h^T ( d h / d A_{ij} ).
    """
    N = A.shape[0]
    # Build Lambda and invert it
    Lambda, v = build_Lambda_and_v(A)
    Lambda_inv = np.linalg.pinv(Lambda)

    # Evaluate the gradient wrt h
    h = np.around(np.array([trophic_levels[node] for node in range(N)]), decimals=14)
    g_h = gradient_incoherence_wrt_h(A, h)  # shape (N,)

    # We'll store partial derivatives in dF_dA
    dF_dA = np.zeros_like(A)
    E = A.sum()

    S = np.sum(A*((h[None, :] - h[:, None]-1)**2))
    # For each A_{ij}, we do the matrix-vector product in a cheap way:
    #   (E_{ij} + E_{ji}) h   =  a vector with:
    #       index i = h[j], index j = h[i], all else = 0
    #
    # Then multiply by Lambda_inv to get d h/dA_{ij}
    # Then dot with g_h
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Build the vector temp = (E_{ij}+E_{ji}) h
            temp = np.zeros(N)
            temp[i] = h[j]
            temp[j] = h[i]
            # plus the partial wrt v
            temp[i] -= 1     # from +Lambda_inv( d v / d A_{ij} )
            temp[j] += 1

            # d h / d A_{ij} = Lambda_inv @ temp
            dh_dAij = Lambda_inv @ temp

            # Finally: dF/dA_{ij} = g_h^T (dh_dAij) + dF_dA
            dF_dA[i,j] = np.dot(g_h, dh_dAij) + (1/E**2) * ((h[j]-h[i]-1)**2 *E - S) if E!= 0 else 0.0


    #If A_ij element was intially zero, set the dA_ij to zero too, and if dA element small, set to zero
    for i in range(N):
        for j in range(N):
          if A[i,j] == 0:
            dF_dA[i,j] = 0

          if abs(dF_dA[i,j]) < 1e-12:
            dF_dA[i,j] = 0

    return dF_dA


def change_incoherence(A, F_target, precision, delta):
  trophic_levels = nobasal_calculate_trophic_levels(A)
  F = nobasal_incoherence(A, trophic_levels)
  dF_dA = gradient_wrt_A(A, trophic_levels, F)
  print('Got the gradient matrix')

  trophic_sequence = [trophic_levels]
  F_sequence = [F]
  adj_sequence = [A]
  attempts = 0

  # Raise or lower incoherence to an initial precision using the gradient map
  initial_precision = precision
  if F > F_target:
    delta = -delta
  while abs(F - F_target) > initial_precision and attempts < 10**5:
    attempts +=1

    # Apply update but ensure nonzero elements don't become zero
    A = np.where(A != 0, np.maximum(A + dF_dA * delta, 1e-3), A)
    trophic_levels = nobasal_calculate_trophic_levels(A)
    F = nobasal_incoherence(A, trophic_levels)

    adj_sequence.append(A)
    trophic_sequence.append(trophic_levels)
    F_sequence.append(F)
    if attempts % 500 == 0: print(F)

  if abs(F - F_target) < initial_precision: print('Precision reached after',attempts,'attempts')

  return A, adj_sequence, trophic_sequence, F_sequence



def evenly_picks(x_vals, y_vals, z_vals, size):
  start = x_vals[0]
  end = x_vals[-1]

  # Create a new, evenly spaced range
  targets = np.linspace(start, end, size)

  used = set()
  filtered_indices = []
  for t in targets:
      # find closest index
      idx = np.argmin(np.abs(np.array(x_vals) - t))
      # if we already used it, skip or do something else
      if idx not in used:
          used.add(idx)
          filtered_indices.append(idx)
      else:
          pass

  selected_x = [x_vals[i] for i in filtered_indices]
  selected_y = [y_vals[i] for i in filtered_indices]
  selected_z = [z_vals[i] for i in filtered_indices]

  return selected_x, selected_y, selected_z



"""Example"""

matrix = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]])

trophic_level = nobasal_calculate_trophic_levels(matrix)
F = nobasal_incoherence(matrix, trophic_level)
print('Initial incoherence F=', F)

A_optimized, A_seq, _, _ = change_incoherence(matrix, F_target= F-0.5, precision=10**-3, delta = 10**-2)
trophic_level_opt = nobasal_calculate_trophic_levels(A_optimized)
F_opt = nobasal_incoherence(A_optimized, trophic_level_opt)
print('New incoherence F=', F_opt)

print('New adjacency matrix', A_optimized)
print(len(A_seq))



"""# Negative weights"""

def weighted_plot(A, hP, hN):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # Get an initial layout (we won't use its coordinates, but its structure can help avoid overlap)
    pos = nx.spring_layout(G, seed=42)

    # Override positions: x-coordinate from hP and y-coordinate from hN
    for node in list(pos.keys()):
        if node not in hP or node not in hN:
            # If a node doesn't have both trophic levels, remove it.
            pos.pop(node)
            continue

        # Set x to hP[node] and y to hN[node]
        pos[node][0] = hN[node]
        pos[node][1] = hP[node]

    # # Rescale the x-coordinates to spread out the nodes horizontally if needed.
    # if pos:
    #     xs = [pos[n][0] for n in pos]
    #     min_x, max_x = min(xs), max(xs)
    #     if max_x > min_x:  # avoid division by zero in trivial graphs
    #         for n in pos:
    #             # Shift to [0, 1] then multiply by xscale.
    #             pos[n][0] = (pos[n][0] - min_x) / (max_x - min_x) * xscale

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='indigo', alpha=0.7)
    # Separate edges based on weight: default weight is assumed positive if not specified.
    edges_neg = [(u, v) for u, v, data in G.edges(data=True) if data.get("weight", 1) >= 0]
    edges_pos = [(u, v) for u, v, data in G.edges(data=True) if data.get("weight", 1) < 0]

    # Draw positive (or nonnegative) weighted edges
    nx.draw_networkx_edges(G, pos, edgelist=edges_pos, alpha=0.3, edge_color='gray', arrows=True)
    # Draw negative weighted edges in pink
    nx.draw_networkx_edges(G, pos, edgelist=edges_neg, alpha=0.3, edge_color='#9370DB', arrows=True)

    max_value = max(max(hP.values(), default=0), max(hN.values(), default=0)) +0.05
    plt.xlim(-0.05, max_value)
    plt.ylim(-0.05, max_value)
    plt.gca().set_aspect('equal')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.ylabel("Trophic levels of positive network")
    plt.xlabel("Trophic levels of negative network")
    plt.grid(False)

    return plt



def weighted_incoherence(A, hP, hN, alpha):
    n = A.shape[0]

    P = np.maximum(A, 0)
    N = np.maximum(-A, 0)

    hP = np.around(np.array([hP[node] for node in range(n)]), decimals=14)
    hN = np.around(np.array([hN[node] for node in range(n)]), decimals=14)

    # Vectorized computation of |s[j] - s[i]| - 1
    # Broadcasting s as a column vector and row vector
    Px_ij = hP[None, :] - hP[:, None]
    Nx_ij = hN[None, :] - hN[:, None]

    Psum = np.sum(P)
    Nsum = np.sum(N)

    PF = np.sum(P * ((Px_ij - 1)**2)) / Psum
    if Nsum == 0:
      NF = 0
      coupling = 0
      print('Network has no negative weights')
    else:
      NF = np.sum(N * ((Nx_ij - 1)**2)) / Nsum
      k = (np.sum((hP - hN)**2)) / n

    F = (1-alpha)*(PF + NF) + alpha*k

    return F, PF, NF, k

def weighted_calculate_trophic_levels(A, alpha):
    n = A.shape[0]

    P = np.maximum(A, 0)
    N = np.maximum(-A, 0)

    Pin_degrees = np.sum(P, axis=0)
    Pout_degrees = np.sum(P, axis=1)
    Pz = Pin_degrees + Pout_degrees
    Pv = (Pin_degrees - Pout_degrees)
    PZ = np.diag(Pz)
    PLambda = PZ - P - np.transpose(P)

    Nin_degrees = np.sum(N, axis=0)
    Nout_degrees = np.sum(N, axis=1)
    Nz = Nin_degrees + Nout_degrees
    Nv = (Nin_degrees - Nout_degrees)
    NZ = np.diag(Nz)
    NLambda = NZ - N - np.transpose(N)

    Alpha = alpha * np.eye(n)
    Ps = np.sum(np.where(P>0, 1, 0))
    Ns = np.sum(np.where(N>0, 1, 0))
    # Reshape V to be (n, 1)
    V = np.concatenate([abs(1-alpha) * Pv/Ps, abs(1-alpha) * Nv/Ns]).reshape(-1, 1)
    # Reshape M to be (n, n)
    M = np.block([[(abs(1-alpha) * PLambda / Ps) + (Alpha /n), -Alpha / n],
                  [-Alpha / n, (abs(1-alpha) * NLambda / Ns) + (Alpha / n)]])

    # Flag to stop the monitoring thread
    stop_monitor = threading.Event()

    def monitor_execution(delay, message):
        if not stop_monitor.wait(delay):  # Wait for delay or an event to stop
            print(message)

    # Start a monitoring thread
    timeout_seconds = 5
    monitor_thread = threading.Thread(target=monitor_execution, args=(timeout_seconds, f"Computation (solving TL eq) is taking longer than {timeout_seconds} seconds..."))
    monitor_thread.start()

    # Solve matrix equation
    try:
        M_inv = np.linalg.pinv(M)
        H = np.dot(M_inv, V)
    except np.linalg.LinAlgError:
        print("Lambda was singular; gave up.")

    # Signal the monitoring thread to stop and wait for it to finish
    stop_monitor.set()
    monitor_thread.join()

    # Extract hs and shift
    Ph = H[:n]
    Nh = H[n:]
    shift = min(min(Ph), min(Nh))
    Ph = Ph - shift
    Nh = Nh - shift

    # Writing trophic levels in dict
    hP = {i: Ph[i][0] for i in range(n)}
    hN = {i: Nh[i][0] for i in range(n)}

    return hP, hN


def negative_mean_activity(A, hP, hN, step, s_tilde, theta):
  N = A.shape[0]
  node_order = range(N)

  # Determine S, the number of sensory nodes
  S = int(round(s_tilde * N))
  S = max(S, 1)  # Ensure at least one node is selected

  # Select sensory nodes: nodes with the lowest trophic levels.
  # (We use the default value 0 if a node's trophic level is not in the dictionary.)
  sensory_nodes_P = sorted(node_order, key=lambda node: hP.get(node, 0))[:S]
  sensory_nodes_N = sorted(node_order, key=lambda node: hN.get(node, 0))[:S]
  sensory_nodes = sensory_nodes_P + sensory_nodes_N

  sig = np.random.choice([-1, 1], size=N).reshape(-1,1)
  states = [dict(zip(node_order, sig.copy()))]

  m = []
  for t in range(step):
    m.append(np.mean(sig))
    new_sig = np.empty_like(sig)
    h = A.T @ sig
    count = 0
    for i in range(N):
      node = node_order[i]
      if node in sensory_nodes:
        h[i] = 0
        new_sig[i] = np.random.choice([-1, 1])
        count += 1
      else:
        proba = 1/2 + np.tanh(h[i]/theta)/2
        r = random.random()
        new_sig[i] = 1 if r < proba else -1
    sig = new_sig
    states.append(dict(zip(node_order, sig.copy())))

  return m, states


"""Example"""
A = np.array([[0, 1, 0, -1],
              [1, 0, 1, -1],
              [0, 1, 0, 0],
              [-1, -1, 0, 0]])

hP, hN = weighted_calculate_trophic_levels(A, 0)
F, PF, NF, coupling = weighted_incoherence(A, hP, hN, 0)
print('Total incoherence F=',F)
print('Incoherence of the positive network Fp=', PF)
print('Incoherence of the negative network Fn=',NF)
print('Coupling strength Fc=',coupling)
plt = weighted_plot(A, hP, hN)
plt.show()

alphas = np.linspace(0, 0.99, 100)
results = []
for alpha in alphas:
  hP, hN = weighted_calculate_trophic_levels(A, alpha)
  F, PF, NF, coupling = weighted_incoherence(A, hP, hN, alpha)
  results.append((F, PF, NF, coupling))

F_values = [res[0] for res in results]
PF_values = [res[1] for res in results]
NF_values = [res[2] for res in results]
coupling_values = [res[3] for res in results]

fig, axes = plt.subplots(2, 2, figsize=(7, 7))
axes[0, 0].plot(alphas, F_values, color='indigo')
axes[0, 0].set_title('H')
axes[0, 1].plot(alphas, PF_values, color='indigo')
axes[0, 1].set_title('$F_P$')
axes[1, 0].plot(alphas, NF_values, color='indigo')
axes[1, 0].set_title('$F_N$')
axes[1, 0].set_xlabel('$\\alpha$')
axes[1, 1].plot(alphas, coupling_values, color='indigo')
axes[1, 1].set_title('$\\kappa$')
axes[1, 1].set_xlabel('$\\alpha$')
plt.tight_layout()
plt.show()



"""Training Hopfield-like models"""

class GraphBasedHopfieldNetwork2:
    """
    Differs from a Hopfield network in that not all nodes are connected to all other nodes: connections are given by input graph.
    There is initial state (given +-1 for each node), and some target patterns. The Hebbian learning rule is applied for each pattern:
    the weight between two connected nodes is increased if their activations are similar.
    Then the majority rule is applied, so that each node's state evolves based on it's weighted connection with other states.

    Args:
      G: An nx graph with N nodes
    Attributes:
      weights: An N x N matrix with the weights between nodes, i.e energy landscape
    Methods:
      train(patterns): Takes a list of N x 1 arrays that the network will remember
      run(initial_state, steps): Takes a N x 1 initial state and a number of steps and evolves the network
    """
    def __init__(self, graph, TL, patterns, beta, gamma):
        self.graph = graph
        self.TLs = TL
        self.patterns = patterns
        self.beta = beta
        self.gamma = gamma
        self.num_neurons = graph.number_of_nodes()
        self.node_order = sorted(self.graph.nodes)
        self.adjacency = nx.to_numpy_array(self.graph, nodelist=self.node_order)

        self.weights = np.zeros((self.num_neurons, self.num_neurons)) # Strength of connection between nodes
        self.weight_list = []
        for pattern in self.patterns:
            # Apply Hebbian learning only for connected nodes
            for i, j in self.graph.edges():
                self.weights[i, j] += pattern[i] * pattern[j] / (2 * len(self.patterns))
                self.weights[j, i] += pattern[i] * pattern[j] / (2 * len(self.patterns))

            adj_weight = self.weights * self.adjacency
            self.weight_list.append(adj_weight)



    def update(self, state, pattern_index, counter, step_show, sensory_nodes):
        new_state = state.copy()
        field = np.sum((self.adjacency.T * self.weights * state), axis=1)
        for i in range(self.num_neurons):
            
          # Get the actual node from the node order
          node = self.node_order[i]
          if counter < step_show and node in sensory_nodes:
            field[i] += self.gamma * self.patterns[pattern_index][i]
            
          # Weighted majority rule
          if field[i] == 0:
            new_state[i] = random.choice([-1,1])
          else:
            proba = 1/2 + np.tanh(field[i]/self.beta)/2
            new_state[i] = 1 if random.random() < proba else -1
            
        return new_state

    def run(self, initial_state, pattern_sequence, step_show, s_tilde):
        "patterns is an array of µ patterns of length N, so order is an array of µ orders, one for each pattern"
        state = initial_state
        patterns = np.array(self.patterns)
        orders = []

        # Determine S, the number of sensory nodes
        S = int(round(s_tilde * self.num_neurons))
        S = max(S, 1)  # Ensure at least one node is selected

        # Select sensory nodes: nodes with the lowest trophic levels.
        # (We use the default value 0 if a node's trophic level is not in the dictionary.)
        sensory_nodes = sorted(self.node_order, key=lambda node: self.TLs.get(node, 0))[:S]

        # Print the highest trophic level among the sensory nodes
        highest_tl = max(self.TLs.get(node, 0) for node in sensory_nodes)
        #print(f"Highest trophic level of sensory nodes: {highest_tl}")

        # Run the network dynamics
        for pattern_index, steps in pattern_sequence:
            for counter in range(int(round(steps))):
                order = np.array([np.dot(state, pattern) / self.num_neurons for pattern in patterns])
                orders.append(order)
                state = self.update(state, pattern_index, counter, step_show, sensory_nodes)
        return state, orders


def performance(orders, steps):
  '''
    orders : an array with each column as the order for a pattern
  '''
  orders = np.array(orders)
  n = orders.shape[1]
  errors = []
  for i in range(n):
    order = np.array(orders[int((i*steps/n)+1):int((i+1)*steps/n), i])
    d = np.mean(1 - order)
    errors.append(d)

  avg_error = np.mean(errors)
  performance = 1 - avg_error
  
  return performance




class GraphBasedHopfieldNetwork:
    """ 
    Same thing but with iterative training
    Args:
        delta: threshold multiplier for the field check
        learning_rate: learning rate for weight updates
        step_max: max number of iterative passes
    """
    def __init__(self,
                 graph,
                 TL,
                 patterns,
                 beta,
                 gamma,
                 delta,       
                 learning_rate,
                 steps_max):
        self.graph = graph
        self.TLs = TL
        self.patterns = patterns
        self.beta = beta
        self.gamma = gamma
        self.num_neurons = graph.number_of_nodes()
        self.node_order = sorted(self.graph.nodes)
        self.delta = delta
        self.learning_rate = learning_rate
        self.steps_max = steps_max

        # Adjacency used for the local field calculation and to mask weight updates
        self.adjacency = nx.to_numpy_array(self.graph, nodelist=self.node_order)

        # Initialize weights to zero
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        # Run iterative Hebb learning
        self.iterative_hebb_learning()

    def iterative_hebb_learning(self):
      self.weight_history = []
      step = 0

      while step < self.steps_max:
          no_update = True

          for pattern in self.patterns:
              # Compute all neurons’ fields at once in a vectorized way.
              field = np.dot((self.adjacency.T * self.weights.T), pattern)

              # Find which neurons fail threshold
              update_indices = np.where(field * pattern < self.delta)[0]

              if update_indices.size > 0:
                # Construct the update matrix (outer product) for just those columns
                update_matrix = self.learning_rate * np.outer(pattern, pattern[update_indices])

                # Apply adjacency mask so only valid edges are updated
                self.weights[:, update_indices] += update_matrix * (self.adjacency[:, update_indices] != 0)

                # Keep a full copy of the weight matrix *after* each bulk update
                self.weight_history.append(self.weights.copy())

                no_update = False

          if no_update:
              # Means no neuron was updated in any pattern, so we are done
              break

          step += 1


    def update(self, state, pattern_index, counter, step_show, sensory_nodes):
        new_state = state.copy()
        field = np.sum((self.adjacency.T * self.weights.T * state), axis=1)

        for i in range(self.num_neurons):
            node = self.node_order[i]

            if counter < step_show and node in sensory_nodes:
                field[i] += self.gamma * self.patterns[pattern_index][i]

            # Weighted majority rule
            if field[i] == 0:
                new_state[i] = random.choice([-1, 1])
            else:
                proba = 0.5 + 0.5 * np.tanh(field[i] / self.beta)
                new_state[i] = 1 if random.random() < proba else -1

        return new_state

    def run(self, initial_state, pattern_sequence, step_show, s_tilde):
        """
        pattern_sequence is a list of (pattern_index, steps) pairs.
        We iterate over them in order, updating the network state.
        """
        state = initial_state
        patterns = np.array(self.patterns)
        orders = []

        # Determine S, the number of sensory nodes
        S = int(round(s_tilde * self.num_neurons))
        S = max(S, 1)  # Ensure at least one node is selected

        # Select sensory nodes: nodes with the lowest trophic levels
        sensory_nodes = sorted(self.node_order, key=lambda node: self.TLs.get(node, 0))[:S]

        for pattern_index, steps in pattern_sequence:
            for counter in range(int(round(steps))):
                # Record overlaps with each stored pattern
                order = np.array([np.dot(state, pattern) / self.num_neurons for pattern in patterns])
                orders.append(order)
                state = self.update(state, pattern_index, counter, step_show, sensory_nodes)

        return state, orders, self.weight_history
    
    
def iterative_pattern_recovery(N, G, TL, beta, gamma, s, n, delta, learning_rate, steps_max, step_show=5, steps=2000, plot=True):
  '''
  To make plots easier
  beta : The temperature for how likely a system is to change its state in the majority rule
  gamma : How strongly the patterns are showed
  s : Proportion of sensory nodes showed
  n : Number of patterns
  step_show : Number of steps a pattern is showed to the basals
  steps : Number of time steps for the hopfield network
  '''

  # A sequence of tuples with the index of the pattern to be showed at at which steps
  pattern_sequence = [(i, steps / n) for i in range(n)]
  patterns = [np.random.choice([1, -1], size=N) for _ in range(n)]
  initial_state = np.random.choice([1, -1], size=N)

  hopfield_net = GraphBasedHopfieldNetwork(G, TL, patterns, beta, gamma, delta, learning_rate, steps_max)
  final_state, orders, weight_list = hopfield_net.run(initial_state, pattern_sequence, step_show, s)

  perf = performance(orders, steps)

  if plot:

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    orders = np.array(orders)

    # Plot the order parameters and save the colors assigned to each pattern
    lines = []
    for i, _ in enumerate(patterns):
        line, = axes.plot(orders[:, i], label=f"Pattern {i + 1}")  # Use `axes`
        lines.append(line)

    # Add horizontal lines for the pattern sequence using the colors from the plotted lines
    current_step = 0
    for pattern_idx, step_duration in pattern_sequence:
        pattern_color = lines[pattern_idx].get_color()  # Get the color assigned to the pattern
        axes.axhline(y=-1, xmin=current_step/steps, xmax=(current_step + step_duration)/steps,
                    color=pattern_color, linewidth=3, alpha=0.6)  # Use `axes`
        current_step += step_duration

    axes.set_ylim(-1.2,1.2)
    axes.set_ylabel("m(t)")
    axes.set_title(f"?")

    plt.tight_layout()
    plt.show()

  return perf, weight_list
   
"""Example"""
graph = nobasal_preferential_preying_model(100, 2000, 2, nobasal_calculate_trophic_levels)
A = nx.to_numpy_array(graph)
TL = nobasal_calculate_trophic_levels(A)
F = nobasal_incoherence(A, TL)
print('F=', F)

steps = 4000 # For the Hopfield network
beta = 0.0001 # The temperature for how likely a system is to change its state in the majority rule
n = 20 # Number of patterns
gamma = 15 # How strongly the patterns are showed
step_show = 5 # Number of steps a pattern is showed to the basals
s = 0.2 # This proportion of nodes will be showed the pattern

delta=1   # threshold multiplier for the field check
learning_rate=1/400   # learning rate for weight updates
steps_max=50

perf, weight_list = iterative_pattern_recovery(100, graph, TL, beta, gamma, s, n, delta, learning_rate, steps_max, step_show=step_show, steps=steps, plot=True)
print('Performance=', perf)
