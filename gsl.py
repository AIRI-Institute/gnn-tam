import torch
import torch.nn as nn
import torch.nn.functional as F


# A = ReLu(W)
class Graph_ReLu_W(nn.Module):
    def __init__(self, n_nodes, k):
        super(Graph_ReLu_W, self).__init__()
        self.num_nodes = n_nodes
        self.k = k
        self.A = nn.Parameter(torch.randn(n_nodes, n_nodes),
                              requires_grad=True)

    def forward(self, idx):
        adj = F.relu(self.A)
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0))
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


# A for Directed graphs:
class Graph_Directed_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k):
        super(Graph_Directed_A, self).__init__()
        self.alpha = alpha
        self.k = k
        self.e1 = nn.Embedding(n_nodes, window_size)
        self.e2 = nn.Embedding(n_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)
        self.l2 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l2(self.e2(idx)))
        adj = F.relu(torch.tanh(self.alpha*torch.mm(m1, m2.transpose(1, 0))))
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0))
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


# A for Uni-directed graphs:
class Graph_Uni_Directed_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k):
        super(Graph_Directed_A, self).__init__()
        self.alpha = alpha
        self.k = k
        self.e1 = nn.Embedding(n_nodes, window_size)
        self.e2 = nn.Embedding(n_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)
        self.l2 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l2(self.e2(idx)))
        adj = F.relu(torch.tanh(self.alpha*(torch.mm(m1, m2.transpose(1, 0))
                                - torch.mm(m2, m1.transpose(1, 0)))))
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0))
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


# A for Undirected graphs:
class Graph_Undirected_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k):
        super(Graph_Directed_A, self).__init__()
        self.alpha = alpha
        self.k = k
        self.e1 = nn.Embedding(n_nodes, window_size)
        self.l1 = nn.Linear(window_size, window_size)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        adj = F.relu(torch.tanh(self.alpha*torch.mm(m1, m2.transpose(1, 0))))
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0))
            mask.fill_(float('0'))
            v, id = (adj + torch.rand_like(adj)*0.01).topk(self.k, 1)
            mask.scatter_(1, id, v.fill_(1))
            adj = adj*mask
        return adj


class Graph_Type_Aware(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, split_by, k_additional):
        super(Graph_Type_Aware, self).__init__()
        self.alpha = alpha
        self.e1 = nn.Embedding(n_nodes+k_additional, window_size)
        self.l1 = nn.Linear(window_size, window_size)
        self.split_by = split_by
        g1 = torch.zeros(n_nodes+k_additional, dtype=int)
        g2 = torch.zeros(n_nodes+k_additional, dtype=int)
        g1[self.split_by[0]] = 1
        g2[self.split_by[1]] = 1
        self.restrictions = nn.Buffer(
            1 - g1[..., None]*g2[None, ...] + g2[..., None]*g1[None, ...]
        )
        self.k_additional = k_additional
        

    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        adj = F.relu(torch.tanh(self.alpha*torch.mm(m1, m2.transpose(1, 0))))
        adj = adj*self.restrictions
        return adj
        




class GSL(nn.Module):
    """
    Graph structure learning block.
    """
    def __init__(
            self,
            gsl_type,
            n_nodes,
            window_size,
            alpha,
            k,
            split_by,
            k_additional,
    ):
        super().__init__()
        self.gsl_layer = None
        if gsl_type == 'relu':
            self.gsl_layer = Graph_ReLu_W(n_nodes, k)
        elif gsl_type == 'directed':
            self.gsl_layer = Graph_Directed_A(n_nodes, window_size,
                                                   alpha, k)
        elif gsl_type == 'unidirected':
            self.gsl_layer = Graph_Uni_Directed_A(n_nodes, window_size,
                                                       alpha, k)
        elif gsl_type == 'undirected':
            self.gsl_layer = Graph_Undirected_A(n_nodes, window_size,
                                                     alpha, k)
        elif gsl_type == 'type_aware':
            self.gsl_layer = Graph_Type_Aware(n_nodes, window_size, alpha, split_by, k_additional)
        else:
            assert False, f'Wrong name of graph structure learning layer: {gsl_type}'

    def forward(self, idx):
        return self.gsl_layer(idx)
