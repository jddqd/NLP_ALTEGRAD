"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.read_edgelist('../datasets/CA-HepTh.txt', comments = '#', delimiter = '\t')
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

############## Task 2

print('Number of connected components : ' + str(nx.number_connected_components(G)))
gcc_nodes = max(nx.connected_components(G), key=len)
gcc = G.subgraph(gcc_nodes)

print(f"Number of nodes in giant component: {gcc.number_of_nodes()}")
print(f"Number of edges in giant component: {gcc.number_of_edges()}")


print(f"Fraction of nodes in giant component: {gcc.number_of_nodes() / G.number_of_nodes()}")
print(f"Fraction of edges in giant component: {gcc.number_of_edges() / G.number_of_edges()}")