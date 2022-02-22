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
from graphgym.models.act import act_dict
from graphgym.contrib.layer.generalconv import (GeneralConvLayer, GeneralEdgeConvLayer)
from graphgym.contrib.layer.attconv import GeneralAddAttConvLayer
from graphgym.contrib.layer.customconv import CustomConvLayer

from graphgym.contrib.layer import *
import graphgym.register as register

## General classes
class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=True, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm and cfg.gnn.l2norm
        has_bn = has_bn and cfg.gnn.batchnorm
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        # print(act_dict.keys)
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)
        return batch


# class CustomGeneralLayer(nn.Module):
#     '''General wrapper for layers with custom message/att/aggr/update func'''

#     def __init__(self, message, att, aggr, update, dim_in, dim_out, has_act=True, has_bn=True,
#                  has_l2norm=False, **kwargs):
#         super(GeneralLayer, self).__init__()
#         self.has_l2norm = has_l2norm
#         has_bn = has_bn and cfg.gnn.batchnorm
#         self.layer = CustomConvLayer(message, att, aggr, update, dim_in, dim_out, bias=not has_bn, **kwargs)
#         # self.layer = layer_dict[name](dim_in, dim_out, bias=not has_bn, **kwargs)
#         layer_wrapper = []
#         if has_bn:
#             layer_wrapper.append(nn.BatchNorm1d(
#                 dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
#         if cfg.gnn.dropout > 0:
#             layer_wrapper.append(nn.Dropout(
#                 p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
#         # print(act_dict.keys)
#         if has_act:
#             layer_wrapper.append(act_dict[cfg.gnn.act])
#         self.post_layer = nn.Sequential(*layer_wrapper)

#     def forward(self, batch):
#         batch = self.layer(batch)
#         if isinstance(batch, torch.Tensor):
#             batch = self.post_layer(batch)
#             if self.has_l2norm:
#                 batch = F.normalize(batch, p=2, dim=1)
#         else:
#             batch.node_feature = self.post_layer(batch.node_feature)
#             if self.has_l2norm:
#                 batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)
#         return batch

class GeneralMultiLayer(nn.Module):
    '''General wrapper for stack of layers'''

    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None,
                 final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


## Core basic layers
# Input: batch; Output: batch 
class _None(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(_None, self).__init__()
        # assert dim_in == dim_out, "in & out dimension not match for none aggr."
        self.model = nn.Linear(dim_in, dim_out, bias=bias)
        self.model.reset_parameters()

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.node_feature = self.bn(batch.node_feature)
        return batch


class BatchNorm1dEdge(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.edge_feature = self.bn(batch.edge_feature)
        return batch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None,
                 num_layers=2, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(
                GeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner,
                                  dim_inner, final_act=True))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch

class GCNConvLayer(pyg.nn.GCNConv):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNConvLayer, self).__init__(dim_in, dim_out, bias=bias)

    def message(self, x_j: Tensor, x_i: OptTensor):
        if cfg.gnn.msg == 'identity':
            return x_j 
        elif cfg.gnn.msg == 'hadamard':
            return x_j * x_i 
        else:
            raise ValueError('cfg.gnn.msg must in [identity, hadamard]')

class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNConv, self).__init__()
        self.model = GCNConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch

# class GCNConv(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super(GCNConv, self).__init__()
#         self.model = pyg.nn.GCNConv(dim_in, dim_out, bias=bias)

#     def forward(self, batch):
#         batch.node_feature = self.model(batch.node_feature, batch.edge_index)
#         return batch
    
class SAGEConvLayer(pyg.nn.SAGEConv):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConvLayer, self).__init__(dim_in, dim_out, bias=bias)

    def message(self, x_j: Tensor, x_i: OptTensor):
        if cfg.gnn.msg == 'identity':
            return x_j 
        elif cfg.gnn.msg == 'hadamard':
            return x_j * x_i 
        else:
            raise ValueError('cfg.gnn.msg must in [identity, hadamard]')

class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = SAGEConvLayer(dim_in, dim_out, bias=bias, **kwargs)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


# class SAGEConv(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super(SAGEConv, self).__init__()
#         self.model = pyg.nn.SAGEConv(dim_in, dim_out, bias=bias, concat=True)

#     def forward(self, batch):
#         batch.node_feature = self.model(batch.node_feature, batch.edge_index)
#         return batch

class GATConvLayer(pyg.nn.GATConv):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConvLayer, self).__init__(dim_in, dim_out, bias=bias)

    def message(self, x_j: Tensor, x_i: OptTensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if cfg.gnn.msg == 'identity':
            return x_j * alpha.unsqueeze(-1)
        elif cfg.gnn.msg == 'hadamard':
            return x_j * x_i * alpha.unsqueeze(-1)
        else:
            raise ValueError('cfg.gnn.msg must in [identity, hadamard]')

    # def message(self, x_j: Tensor, x_i: OptTensor):
    #     if cfg.gnn.msg == 'identity':
    #         return x_j 
    #     elif cfg.gnn.msg == 'hadamard':
    #         return x_j * x_i 
    #     else:
    #         raise ValueError('cfg.gnn.msg must in [identity, hadamard]')

class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = GATConvLayer(dim_in, dim_out, bias=bias, **kwargs)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch

# class GATConv(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super(GATConv, self).__init__()
#         self.model = pyg.nn.GATConv(dim_in, dim_out, bias=bias)

#     def forward(self, batch):
#         batch.node_feature = self.model(batch.node_feature, batch.edge_index)
#         return batch

class GINConvLayer(pyg.nn.GINConv):
    def __init__(self, nn, **kwargs):
        super(GINConvLayer, self).__init__(nn)

    def message(self, x_j: Tensor, x_i: OptTensor):
        if cfg.gnn.msg == 'identity':
            return x_j 
        elif cfg.gnn.msg == 'hadamard':
            return x_j * x_i 
        else:
            raise ValueError('cfg.gnn.msg must in [identity, hadamard]')

class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))
        self.model = GINConvLayer(gin_nn)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


# class GINConv(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super(GINConv, self).__init__()
#         gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
#                                nn.Linear(dim_out, dim_out))
#         self.model = pyg.nn.GINConv(gin_nn)

#     def forward(self, batch):
#         batch.node_feature = self.model(batch.node_feature, batch.edge_index)
#         return batch


# class SplineConv(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super(SplineConv, self).__init__()
#         self.model = pyg.nn.SplineConv(dim_in, dim_out,
#                                        dim=1, kernel_size=2, bias=bias)

#     def forward(self, batch):
#         batch.node_feature = self.model(batch.node_feature, batch.edge_index,
#                                         batch.edge_feature)
#         return batch


class GeneralConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


class GeneralSampleEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < cfg.gnn.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_feature[edge_mask, :]
        batch.node_feature = self.model(batch.node_feature, edge_index,
                                        edge_feature=edge_feature)
        return batch


layer_dict = {
    'none': _None,
    'linear': Linear,
    'mlp': MLP,
    'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    # 'splineconv': SplineConv,
    'ginconv': GINConv,
    'generalconv': GeneralConv,
    'generaledgeconv': GeneralEdgeConv,
    'generalsampleedgeconv': GeneralSampleEdgeConv
}

# register additional convs
layer_dict = {**register.layer_dict, **layer_dict}
