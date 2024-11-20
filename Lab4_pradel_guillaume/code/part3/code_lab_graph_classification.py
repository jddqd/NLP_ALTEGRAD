"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import torch


############## Task 7


#load Mutag dataset
def load_dataset():

    dataset = TUDataset(root = '../datasets/',name = 'MUTAG')


    Gs = []
    for data in dataset:
        G = to_networkx(data, to_undirected=True)
        Gs.append(G)
    
    y = dataset.data.y.numpy()

    return Gs, y


Gs,y = load_dataset()

G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)


# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))


    for i in range(len(G_train)):
        for j in range(n_samples):
            
            set_nodes = np.random.choice(G_train[i].nodes(),3,replace=False)
            sub = G_train[i].subgraph(set_nodes)

            if nx.is_isomorphic(sub,graphlets[0]):
                phi_train[i][0] += 1
            elif nx.is_isomorphic(sub,graphlets[1]):
                phi_train[i][1] += 1
            elif nx.is_isomorphic(sub,graphlets[2]):
                phi_train[i][2] += 1
            else :
                phi_train[i][3] += 1
        

    phi_test = np.zeros((len(G_test), 4))
    

    for i in range(len(G_test)):
        for j in range(n_samples):
        
            set_nodes = np.random.choice(G_test[i].nodes(),3,replace=False)
            sub = G_test[i].subgraph(set_nodes)

            if nx.is_isomorphic(sub,graphlets[0]):
                phi_test[i][0] += 1
            elif nx.is_isomorphic(sub,graphlets[1]):
                phi_test[i][1] += 1
            elif nx.is_isomorphic(sub,graphlets[2]):
                phi_test[i][2] += 1
            else :
                phi_test[i][3] += 1

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

############## Task 9


K_train_gp, K_test_gp = graphlet_kernel(G_train, G_test)

############## Task 10

# SVM for graphlet

clf_graphlet = SVC(kernel='precomputed')
clf_graphlet.fit (K_train_gp, y_train)
ypred_graphlet = clf_graphlet.predict( K_test_gp)

# SVM for shortestpath

clf_sp = SVC(kernel='precomputed')
clf_sp.fit (K_train_sp, y_train)
ypred_sp = clf_sp.predict( K_test_sp)



print("Accuracy using shortest path kernel : " + str(accuracy_score(y_test,ypred_sp) * 100) + "%")

print("Accuracy using graphlet kernel : " + str(accuracy_score(y_test,ypred_graphlet) * 100) + "%")

