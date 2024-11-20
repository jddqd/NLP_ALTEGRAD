"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    A = nx.adjacency_matrix(G)
    D = diags([1/G.degree(node) for node in G.nodes()])
    L = np.eye(G.number_of_nodes()) - D @ A

    eigenvalues, eigenvectors = eigs(L, k=k, which='SR')
    eigenvectors = eigenvectors.real

    kmeans= KMeans(n_clusters=k)
    kmeans.fit(eigenvectors)
    clustering = {}
    for i,node in enumerate(G.nodes()):
      clustering[node] = kmeans.labels_[i]  
    
    
    return clustering


############## Task 4

G = nx.read_edgelist('../datasets/CA-HepTh.txt', comments = '#', delimiter = '\t')
gcc_nodes = max(nx.connected_components(G), key=len)
gcc = G.subgraph(gcc_nodes)

clustering = spectral_clustering(gcc, 50)
# print(clustering)


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    

    m = G.number_of_edges()
    clusters = set(clustering.values())
    modularity = 0

    for cluster in clusters:

      nodes = [node for node in G.nodes() if clustering[node] == cluster]
      sub = G.subgraph(nodes)
      lc = sub.number_of_edges()

      dc = 0
      for node in nodes:
        dc += G.degree(node)

      modularity += (lc/m) - ((dc/m)**2) 
    
    
    
    return modularity



############## Task 6


print(modularity(gcc, clustering))

random_clustering = {}
for node in gcc.nodes():
    random_clustering[node] = randint(0, 49)

print(modularity(gcc, random_clustering))





