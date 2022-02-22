from numpy import isin, mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import kaiming_uniform_


from graphgym.config import cfg
from graphgym.models.head import head_dict
from graphgym.models.layer import (GeneralLayer, GeneralMultiLayer,
                                   BatchNorm1dNode, BatchNorm1dEdge)
import graphgym.register as register
from graphgym.models.act import act_dict
from graphgym.models.feature_augment import Preprocess
from graphgym.init import init_weights
from graphgym.models.feature_encoder import node_encoder_dict, edge_encoder_dict

from graphgym.contrib.stage import *

from gpu_mem_track import MemTracker

gpu_tracker = MemTracker()


########### Layer ############
def GNNLayer(dim_in, dim_out, has_act=True):
    return GeneralLayer(cfg.gnn.layer_type, dim_in, dim_out, has_act)


def GNNPreMP(dim_in, dim_out):
    return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                             dim_in, dim_out, dim_inner=dim_out, final_act=True)


########### Block: multiple layers ############

class GNNSkipBlock(nn.Module):
    '''Skip block for GNN'''

    def __init__(self, dim_in, dim_out, num_layers, is_lastblock=False):
        super(GNNSkipBlock, self).__init__()
        self.is_lastblock = is_lastblock 
        if num_layers == 1:
            self.f = [GNNLayer(dim_in, cfg.gnn.dim_inner, has_act=False)]
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(d_in, dim_out))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(d_in, dim_out, has_act=False))
        self.f = nn.Sequential(*self.f)
        self.act = act_dict[cfg.gnn.act]
        # self.act = act_dict['tanh']
        if cfg.gnn.stage_type == 'skipsum' or cfg.gnn.stage_type == 'skipmean':
            assert dim_in == dim_out, 'Sum/Mean skip must have same dim_in, dim_out'

    def forward(self, batch):
        node_feature = batch.node_feature
        if cfg.gnn.stage_type == 'skipsum' or cfg.gnn.stage_type == 'skipmean':
            batch.node_feature = \
                node_feature + self.f(batch).node_feature
        elif cfg.gnn.stage_type == 'skipconcat':
            # input & output of the last block will not be concat to unify the out_dim
            # if not self.is_lastblock:
            batch.node_feature = \
            torch.cat((node_feature, self.f(batch).node_feature), 1)
            # else:
            #     batch = self.f(batch)
        else:
            raise ValueError('cfg.gnn.stage_type must in [skipsum, skipmean, skipconcat]')
        batch.node_feature = self.act(batch.node_feature)
        return batch


########### Stage: NN except start and head ############

class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch

class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipStage, self).__init__()
        assert num_layers % cfg.gnn.skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        self.num_layers = num_layers
        for i in range(num_layers // cfg.gnn.skip_every):
            if cfg.gnn.stage_type == 'skipsum' or cfg.gnn.stage_type == 'skipmean':
                d_in = dim_in if i == 0 else dim_out
            elif cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
                d_out = d_in + dim_out
            if i == num_layers // cfg.gnn.skip_every - 1:
                block = GNNSkipBlock(d_in, d_out, cfg.gnn.skip_every, is_lastblock=True)
            else:
                block = GNNSkipBlock(d_in, d_out, cfg.gnn.skip_every)
            self.add_module('block{}'.format(i), block)
        # if cfg.gnn.stage_type == 'skipconcat':
        #     self.dim_out = d_in + dim_out
        # else:
        #     self.dim_out = dim_out
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.stage_type == 'skipmean':
            # plus layer 0
            node_feature = batch.node_feature
            batch.node_feature = node_feature / (self.num_layers // cfg.gnn.skip_every + 1)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
    'skipmean': GNNSkipStage
}

# Merge 2 dicts 
stage_dict = {**register.stage_dict, **stage_dict}


########### Model: start + stage + head ############

class GNN(nn.Module):
    '''General GNN model'''

    def __init__(self, dim_in, dim_out, **kwargs):
        """
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(GNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        # GNNStage = stage_dict[cfg.gnn.stage_type]
        # GNNHead = head_dict[cfg.dataset.task]

        self.node_encoder = []
        if cfg.dataset.node_encoder:
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim, num_classes=dim_in)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            # Update dim_in to reflect the new dimension fo the node features
            dim_in = cfg.dataset.encoder_dim
        
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)
        
        if cfg.dataset.augment_feature != []:
            self.preprocess = Preprocess(dim_in)
            d_in = self.preprocess.dim_out
        else:
            d_in = cfg.dataset.encoder_dim

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        else:
            d_in = cfg.dataset.encoder_dim

        if cfg.gnn.layers_mp > 0:
            for i in range(cfg.gnn.layers_mp):
                layer = GNNLayer(dim_in=d_in, dim_out=cfg.gnn.dim_inner)
                self.add_module('GNN_Layer_{}'.format(i), layer)

        # if cfg.gnn.layers_mp > 0:
        #     self.mp = GNNStage(dim_in=d_in,
        #                        dim_out=cfg.gnn.dim_inner,
        #                        num_layers=cfg.gnn.layers_mp)
            # d_in = self.mp.dim_out

        # self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        gnn_emb_list = []
        for module in self.children():
            if isinstance(module, GeneralLayer) and len(gnn_emb_list) == 0:
                gnn_emb_list.append(batch.node_feature)
            batch = module(batch)
            if isinstance(module, GeneralLayer):
                gnn_emb_list.append(batch.node_feature)

        node_feature = torch.zeros_like(gnn_emb_list[0])
        if cfg.gnn.stage_type == 'stack':
            pass
        elif cfg.gnn.stage_type == 'sum' or cfg.gnn.stage_type == 'mean':
            for e in gnn_emb_list:
                node_feature = node_feature + e
            if cfg.gnn.stage_type == 'mean':
                node_feature = node_feature / (cfg.gnn.layers_mp + 1)
            batch.node_feature = node_feature
        elif cfg.gnn.stage_type == 'concat':
            node_feature = torch.cat(gnn_emb_list, dim=-1)
            batch.node_feature = node_feature
        else:
            raise(ValueError('cfg.gnn.stage_type must in [stack, sum, mean, concat]'))
        return batch

class MCGNN(nn.Module):
    '''GNN with multiple components'''
    def __init__(self, dim_in, dim_out, **kwargs):
        '''
        dim_in/out: in/out dims of GNN
        com_num: # of components
        '''
        super(MCGNN, self).__init__()
        GNNHead = head_dict[cfg.dataset.task]

        self.dim_in = dim_in
        if cfg.gnn.stage_type == 'concat':
            self.dim_gnn_out = cfg.gnn.dim_inner * (cfg.gnn.layers_mp + 1)
        else:
            self.dim_gnn_out = cfg.gnn.dim_inner
        self.dim_out = dim_out # 1

        for i in range(cfg.gnn.component_num):
            gnn = GNN(dim_in=self.dim_in, dim_out=self.dim_gnn_out)
            self.add_module('GNN_{}'.format(i), gnn)
        
        self.lin = nn.Linear(self.dim_gnn_out * cfg.gnn.component_num, self.dim_gnn_out)
        self.att1 = nn.Linear(self.dim_gnn_out * cfg.gnn.component_num, self.dim_gnn_out)
        self.att2 = nn.Linear(self.dim_gnn_out, cfg.gnn.component_num)
        self.softmax = nn.Softmax()

        self.post_mp = GNNHead(dim_in=self.dim_gnn_out, dim_out=self.dim_out)

        self.reset_parameters()
 
    def reset_parameters(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


    def forward(self, batch):
        # list embeddings of different components
        emb_list = []
        for module in self.children():
            if isinstance(module, GNN):
                batch = module(batch)
                emb_list.append(batch.node_feature)
                batch.node_feature = None
        
        if cfg.gnn.component_aggr == 'mean':
            batch.node_feature = torch.zeros_like(emb_list[0])
            for e in emb_list:
                batch.node_feature = batch.node_feature + e
            batch.node_feature = batch.node_feature / cfg.gnn.component_num
        elif cfg.gnn.component_aggr == 'concat' or cfg.gnn.component_aggr == 'att':
            x = torch.cat(emb_list, dim=-1)
            if cfg.gnn.component_aggr == 'concat':
                batch.node_feature = self.lin(x)
            else:
                x = F.relu(self.att1(x), inplace = True)
                x = self.att2(x)

                att_w = F.softmax(x, dim = 1)
                att_w_list = att_w.chunk(cfg.gnn.component_num, dim = 1)
                # for a in att_w_list:
                #     a.repeat(self.dim_gnn_out, 1)

                for i in range(cfg.gnn.component_num):
                    if i == 0:
                        batch.node_feature = torch.mul(emb_list[i], att_w_list[i])
                    else:
                        batch.node_feature = batch.node_feature + torch.mul(emb_list[i], att_w_list[i])
        else:
            raise ValueError('cfg.gnn.component_aggr must in [mean, concat, att]')

        # If model is being trained, return the output after GNNHead; 
        # else, return original graph batch
        if self.training or cfg.model.eval_type != 'ranking':
            pred, true = self.post_mp(batch)
            return pred, true
        else:
            rating_matrix = self.post_mp(batch)
            return rating_matrix
        # batch = self.post_mp(batch)
        # return batch
