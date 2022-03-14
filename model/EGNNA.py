import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import GraphAttentionWithEdgeConcat, GraphConvolution
from utils import train_utils


class ConcatDecoder(nn.Module):
    """Decoder for using MLP for prediction."""

    def __init__(self, dropout, n_node_features, hidden_1, hidden_2, n_edge_features, n_classes=1):
        super(ConcatDecoder, self).__init__()
        self.dropout = dropout
        self.n_classes = n_classes
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, z, raw_adj_list):
        # z : (N, M), raw_adj_list : [(N, M)] * P
        N = raw_adj_list[0].shape[0]
        z = F.dropout(z, self.dropout, training=self.training)
        raw_adj_list = [adj.view(N, N, 1) for adj in raw_adj_list] # [(N, M, 1)] * P
        raw_adj_list = torch.cat(raw_adj_list, dim=2) # (N, M, P)
        z1 = z.repeat(1, N).view(N, N, -1)
        z2 = z.repeat(N, 1).view(N, N, -1)
        cos_ = self.cos(z1, z2).reshape(N * N, 1)
        adj = torch.cat([z1.view(N * N, -1), z2.view(N * N, -1)], dim=1).view(N, N, -1) # (N, N, 2M)
        del z1, z2
        adj = torch.cat([adj, raw_adj_list], dim=2) # (N, M, 2M+P)
        del z, raw_adj_list
        return adj, cos_ # (N, M, 2M+P)


class MLP(nn.Module):
    """Decoder for using MLP for prediction."""

    def __init__(self, dropout, n_node_features, hidden_1, hidden_2, n_edge_features, n_classes=1):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.n_classes = n_classes
        n_input = n_node_features * 2 + n_edge_features
        self.fc1 = nn.Linear(n_input, hidden_1) # (2M+P, M)
        self.fc2 = nn.Linear(hidden_1, hidden_2) # (M, M)
        self.fc3 = nn.Linear(hidden_2, n_classes)  # (M, 2)

    def forward(self, adj):
        adj = F.relu(self.fc1(adj), inplace=True) # (N, N, M)
        adj = F.dropout(adj, self.dropout, training=self.training)
        frame = inspect.currentframe()
        adj = F.relu(self.fc2(adj), inplace=True) # (N, N, M)
        adj = F.dropout(adj, self.dropout, training=self.training)
        adj = self.fc3(adj) # (N, N, 2)
        adj = F.log_softmax(adj, dim=2).view(-1, self.n_classes)
        return adj # (N, N, 2)


class EGNNA(nn.Module):
    def __init__(self, n_node_features, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                 n_classes, n_edge_features, dropout=0.1):
        #  n_node_features : M, n_hidden_1 : M, n_hidden_2 : M, n_hidden_3 : M, n_hidden_4 : M, n_classes : 2
        super(EGNNA, self).__init__()
        self.dropout = dropout
        self.gcn1 = GraphAttentionWithEdgeConcat(n_node_features, n_hidden_1, n_edge_features) # (N, M)
        self.gcn2 = GraphAttentionWithEdgeConcat(n_hidden_1, n_hidden_2, n_edge_features) # (N, M)
        # self.gcn = GraphConvolution(n_node_features, n_hidden_1)
        self.decoder = ConcatDecoder(dropout, n_hidden_2, n_hidden_3, n_hidden_4, n_edge_features, 
                                     n_classes=n_classes)
        self.mlp = MLP(dropout, n_hidden_2, n_hidden_3, n_hidden_4, n_edge_features, n_classes=n_classes)

    def encode(self, x, adj_list):
        # x = (N, M), adj_list = [(N, N) * P]
        # for adj in adj_list:
        #     x = torch.mul(torch.tanh(self.gcn(x, adj)), x)
        y = F.relu(self.gcn1(x, adj_list), inplace=True)
        # y = self.gcn1(x, adj_list)
        z = self.gcn1(y, adj_list)
        x = (y + z)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn2(x, adj_list)
        del y, z
        return x

    def decode(self, *args):
        # *args = (N, M) , [(N, N) * P]
        adj = self.decoder(*args)
        return adj

    def forward(self, x, adj_list, raw_adj_list):
        # x = (N, M), adj_list = [(N, N) * P], raw_adj_list = [(N, N) * P]
        x = x.squeeze()
        adj_list = list(map(lambda x: x.squeeze(0), adj_list))
        raw_adj_list = list(map(lambda x: x.squeeze(0), raw_adj_list))
        embedding = self.encode(x, adj_list) # (N, M)
        raw_adj_list = train_utils.cuda_list_object_1d(raw_adj_list)
        del x
        del adj_list
        output, cos_ = self.decode(embedding, raw_adj_list) # (N, N, 2M+P)
        del embedding
        del raw_adj_list
        output = self.mlp(output) # (N, N, 2)
        return output, cos_