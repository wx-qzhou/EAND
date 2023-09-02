import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import inspect

class Full_Connection(nn.Module):

    def __init__(self, n_input, hidden, bias=True):
        super(Full_Connection, self).__init__()
        self.n_input = n_input
        self.hidden = hidden
        self.weight = Parameter(torch.FloatTensor(n_input, hidden)) # (M, M)
        if bias:
            self.bias = Parameter(torch.FloatTensor(hidden)) # (M)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        output = torch.matmul(adj, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output # (N, M)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q : (H, N, M), k : (H, N, M), v : (H, N, M)
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2)) # (H, N, M) * (H, M, N) -> (H, N, N)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        # attn = F.relu(attn, inplace=True)
        output = torch.matmul(attn, v) # (H, N, M)

        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, d_k, d_v, n_head=1, dropout=0.1): #n_head=1 dropout=0.1
        # n_head : H, d_model : M, d_k : K, d_v : V, K==V==M
        super().__init__()

        self.n_head = n_head # the number of heads
        self.d_k = d_k # the dim of keys
        self.d_v = d_v # the dim of values

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) # the weight of queries (M, H*K)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False) # the weight of keys    (M, H*K)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False) # the weight of values  (M, H*V)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False) # the weight of FC        (H*V, M)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.leaky_relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        # q : (N, M), k : (N, M), v : (N, M)
        len_q, len_k, len_v = q.size(0), k.size(0), v.size(0) # N, N, N, N

        residual = q
        # K = M
        q = self.w_qs(q).view(len_q, self.n_head, self.d_k) # (N, H*K) -> (N, H, M)
        k = self.w_ks(k).view(len_k, self.n_head, self.d_k) # (N, H*K) -> (N, H, M)
        v = self.w_vs(v).view(len_v, self.n_head, self.d_v) # (N, H*K) -> (N, H, M)

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1) # (H, N, M)

        if mask is not None:
            mask = mask.repeat(self.n_head, 1).view(self.n_head, len_q, -1)   # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask) # (H, N, M)
        q = q.transpose(0, 1).contiguous().view(len_q, -1) # (H, N, M) -> (H, N*M)
        q = self.fc(q)
        q = self.dropout(q) # (N, H*M) * (H*M, M) -> (N, M)
        
        del k, v
        
        # q = self.leaky_relu(q)
        q = F.relu(q, inplace=True)
        q = q + residual

        # q = self.layer_norm(q) # (N, M)

        return q

class GraphAttention(Module):
    def __init__(self, in_features, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.W = Parameter(torch.FloatTensor(in_features, in_features)) # (N, P)
        self.weight_att = Parameter(torch.FloatTensor(2 * in_features, 1)) # (2*M, 1)
        self.reset_parameters()
        self.leaky_relu = nn.LeakyReLU()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.weight_att.data.uniform_(-stdv, stdv)

    def forward(self, node, adj):
        # node : (N, M); adj : (N, N)
        N = node.size()[0]
        h = torch.matmul(node, self.W)

        z = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, -1) # (N, M)||(N, M) -> (N, N, 2M)
        z_att = self.leaky_relu(torch.matmul(z, self.weight_att).squeeze(2)) # (N, N, 2M) * (2M, 1) -> (N, N, 1) ->(N, N)
        # z_att = F.relu(torch.matmul(z, self.weight_att).squeeze(2), inplace=True)
        del z
        zero_vec = -9e15*torch.ones(1)
        zero_vec = zero_vec.cuda()
        z_att = torch.where(adj > 0, z_att, zero_vec) # (N, N)
        z_att = F.softmax(z_att, dim=1)
        z_att = F.dropout(z_att, self.dropout, training=self.training) # (N, N)
        z_att = torch.matmul(z_att, node) # (N, M)
        # z_att = F.relu(z_att, inplace=True)

        node = h + z_att

        del z_att, adj, zero_vec, h

        return node

class GraphAttentionWithEdgeConcat(Module):

    def __init__(self, in_features, out_features, edge_features, bias=True, dropout=0.3):
        super(GraphAttentionWithEdgeConcat, self).__init__()
        self.in_features = in_features # M
        self.out_features = out_features # M
        self.edge_features = edge_features # L
        self.dropout = dropout

        self.leaky_relu = nn.LeakyReLU()
        self.atten = MultiHeadAttention(self.in_features, self.in_features, self.in_features)
        self.graphatten = GraphAttention(self.in_features)

        self.weight_at  = Parameter(torch.FloatTensor(3 * in_features, in_features)) # (2*M, 1)
        self.weight = Parameter(torch.FloatTensor(in_features * edge_features, out_features)) # (M*L, M)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)) # (M)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight_at.data.uniform_(-stdv, stdv)
    
    def normalize(self, mx):
        rowsum = mx.sum(dim=1)
        r_inv = (rowsum ** -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv).to_sparse()
        mx = torch.spmm(r_mat_inv, mx.squeeze())
        del rowsum, r_inv, r_mat_inv
        return mx

    def node_attention(self, node, adj):
        # node : (N, M); adj : (N, N)
        z_att = self.graphatten(node, adj) # (N, M)

        mask = torch.where(adj == 0, adj, torch.ones(1).cuda()) # # (N, N)
        mask = mask.byte()
        z_at = self.atten(node, node, node, mask)
        
        z_a = torch.cat([z_att, z_at, node], dim=1) # (N, 3 * M)
        z_a = torch.matmul(z_a, self.weight_at)
        z_a = self.leaky_relu(z_a)
        # z_a = F.relu(z_a, inplace=True)

        node = z_att + z_a
        
        del adj, z_att, mask, z_a
        
        return node # (N, M)
    
    def adj_attention(self, adj):
        # adj : (N, N)
        adj_att = torch.mul(F.softmax(adj, dim=1), adj)
        adj = adj_att + adj # (N, N)
        del adj_att
        return adj # (N, N)

    def forward_(self, node, adj):
        # node : (N, M); adj : (N, N)
        node = self.node_attention(node, adj) # (N, M)
        adj = self.adj_attention(adj) # (N, N) 
        node = torch.spmm(adj, node) # (N, M)
        del adj
        return node # (N, M)

    def forward(self, node, adj_list):
        # node : (N, M); adj_list : [(N, M)] * P
        support_list = []
        for adj in adj_list:
            support_list.append(self.forward_(node, adj)) # [(N, M)] * P
        support = torch.cat(support_list, dim=1) # (N, M * P)
        output = torch.mm(support, self.weight) # (N, M)
        del node
        del adj_list
        del support
        del support_list
        if self.bias is not None:
            return output + self.bias
        else:
            return output # (N, M)

"""Simple GCN layer, similar to https://arxiv.org/abs/1609.02907"""
class GraphConvolution(nn.Module):

    def __init__(self, in_feature_size, out_feature_size, act=F.relu, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_feature_size = in_feature_size
        self.out_feature_size = out_feature_size
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_feature_size, out_feature_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node, adj):
        support = torch.mm(node, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        del node, adj, support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def description(self):
        print("This is a graph convolution nerul network.")
