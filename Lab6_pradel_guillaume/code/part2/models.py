"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy as sp
from utils import sparse_mx_to_torch_sparse_tensor
from torch_geometric.nn import aggr

# sum_aggr = aggr.SumAggregation()

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()



    def forward(self, x_in, adj, idx):

        ############## Task 6
    
        adj = adj + sparse_mx_to_torch_sparse_tensor(sp.sparse.identity(adj.shape[0]))
        x = self.relu(adj @ self.fc1(x_in))
        x = self.relu(adj @ self.fc2(x))

        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(self.device)
        out = out.scatter_add_(0, idx, x) 

        x = self.fc3(out)
        x = self.relu(x)
        out = self.fc4(x)

        return F.log_softmax(out, dim=1)
