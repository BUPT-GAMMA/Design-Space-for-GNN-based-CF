from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add

import torch_geometric as pyg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, softmax
from torch_geometric.nn.inits import glorot

from graphgym.config import cfg
from graphgym.register import register_layer

class CustomConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout: float=0.0, cached=True, **kwargs):
        if cfg.gnn.agg == 'add' or cfg.gnn.agg == 'mean':
            super(CustomConvLayer, self).__init__(aggr=cfg.gnn.agg)
        elif cfg.gnn.agg == 'none':
            super(CustomConvLayer, self).__init__()
        else:
            raise ValueError('cfg.gnn.agg must in [add, mean, none]')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.dropout = dropout
        self.normalize = cfg.gnn.normalize_adj
        self.add_self_loop = cfg.gnn.add_self_loop

        self.lin_weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.lin_weight2 = Parameter(torch.Tensor(2 * self.in_channels, self.out_channels))

        if cfg.gnn.att is True:
            self.att_l = Parameter(torch.Tensor(1, self.out_channels))
            self.att_r = Parameter(torch.Tensor(1, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            glorot(param)
        # glorot(self.lin_weight)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None):
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.lin_weight)

        if cfg.gnn.agg == 'none':
            return x

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)

        if self.add_self_loop:
            fill_value = 1
            num_nodes = x.size(0)
            edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(0), edge_weight)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        alpha_l = alpha_r = None

        if cfg.gnn.att is True:
            assert x.dim() == 2, 'Static graphs not supported in `attention`'
            alpha_l = (x * self.att_l).sum(dim=-1).view(-1, 1)
            alpha_r = (x * self.att_r).sum(dim=-1).view(-1, 1)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, norm=norm, alpha=(alpha_l, alpha_r))

        x_r = x[1]
        if cfg.gnn.combine == 'identity':
            pass
        elif cfg.gnn.combine == 'add':
            out += x_r
        elif cfg.gnn.combine == 'concat':
            out = torch.cat((x_r, out), dim=1)
            # self.lin_weight2 = Parameter(torch.Tensor(2 * self.out_channels, self.out_channels))
            # glorot(self.lin_weight2)
            out = torch.matmul(out, self.lin_weight2)

        # if self.normalize:
        #     out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, x_i: OptTensor, norm: OptTensor, alpha_j: OptTensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        if alpha_j is None:
            alpha = torch.ones((x_j.size(0),), dtype=x_j.dtype, device=x_j.device).view(-1, 1)
            # alpha = torch.ones_like(x_j)
        elif alpha_i is None:
            alpha = alpha_j
        else:
            alpha = alpha_i + alpha_j
        if alpha_j is not None:
            alpha = F.relu(alpha, inplace=cfg.mem.inplace)
            alpha = softmax(alpha, index, ptr, size_i)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if cfg.gnn.msg == 'identity':
            return norm.view(-1, 1) * x_j * alpha if norm is not None else x_j * alpha
        elif cfg.gnn.msg == 'hadamard':
            return norm.view(-1, 1) * x_j * x_i * alpha if norm is not None else x_j * x_i * alpha
        else:
            raise ValueError('cfg.gnn.msg must in [identity, hadamard]')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, msg={cfg.gnn.msg}, att={cfg.gnn.att}, agg={cfg.gnn.agg}, combine={cfg.gnn.combine})'


class CustomConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(CustomConv, self).__init__()
        self.model = CustomConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


register_layer('customconv', CustomConv)
