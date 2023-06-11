import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add

from torch_geometric.graphgym.register import *


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out








class MultiHeadAttentionLayerSANSparse(nn.Module):
    """Multi-Head Graph Attention Layer.
        Scaled Dot-product
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=None, dropout=0., act=None,
                 edge_enhance=False,
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        # if self.edge_enhance:
        #     self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
        #     nn.init.xavier_normal_(self.VeRow)


    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]     # (num real edges) x num_heads x out_dim
        score = src * dest / np.sqrt(self.out_dim)                   # element-wise multiplication


        # Use available edge features to modify the scores for edges
        if batch.get("E", None) is not None:
            E_w = batch.E.view(-1, self.num_heads, self.out_dim)
            # (num real edges) x num_heads x out_dim
            score = score * E_w

        e_t = score
        score = score.sum(dim=-1, keepdim=True)
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])  # (num real edges) x num_heads x 1
        score = self.dropout(score)
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.attn = score # fixme: for case study only

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        # if self.edge_enhance and batch.E is not None:
        #     rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
        #     rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
        #     batch.wV = batch.wV + rowV


    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)



        return h_out, e_out


class MultiHeadAttentionLayerGraphormerSparse(nn.Module):
    """Multi-Head Graph Attention Layer.
        Scaled Dot-product
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=None, dropout=0., act=None,
                 edge_enhance=False,
                 **kwargs):
        super().__init__()

        clamp = None

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)


    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]     # (num real edges) x num_heads x out_dim
        score = src * dest / np.sqrt(self.out_dim)                   # element-wise multiplication
        score = score.sum(dim=-1, keepdim=True)


        # Use available edge features to modify the scores for edges
        if batch.get("E", None) is not None:
            E_b = batch.E.view(-1, self.num_heads, 1)
            # (num real edges) x num_heads x out_dim
            score = score + E_b

        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])  # (num real edges) x num_heads x 1
        score = self.dropout(score)
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]

        batch.attn = score # fixme: for case study only

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        # if self.edge_enhance and batch.E is not None:
        #     rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
        #     rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
        #     batch.wV = batch.wV + rowV


    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out
