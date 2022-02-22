""" GNN heads are the last layer of a GNN right before loss computation.

They are constructed in the init function of the gnn.GNN.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from graphgym.config import cfg
from graphgym.models.layer import MLP
from graphgym.models.pooling import pooling_dict

from graphgym.contrib.head import *
import graphgym.register as register

from gpu_mem_track import MemTracker

# gpu_tracker = MemTracker()


########### Head ############

class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''

    def __init__(self, dim_in, dim_out):
        super(GNNNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in, dim_out,
                                 num_layers=cfg.gnn.layers_post_mp, bias=True)

    def _apply_index(self, batch):
        if batch.node_label_index.shape[0] == batch.node_label.shape[0]:
            return batch.node_feature[batch.node_label_index], batch.node_label
        else:
            return batch.node_feature[batch.node_label_index], \
                   batch.node_label[batch.node_label_index]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


class GNNEdgeHead(nn.Module):
    '''Head of GNN, edge prediction'''

    def __init__(self, dim_in, dim_out):
        ''' Head of Edge and link prediction models.

        Args:
            dim_out: output dimension. For binary prediction, dim_out=1.
        '''
        # Use dim_in for graph conv, since link prediction dim_out could be
        # binary
        # E.g. if decoder='dot', link probability is dot product between
        # node embeddings, of dimension dim_in
        super(GNNEdgeHead, self).__init__()
        # module to decode edges from node embeddings

        if cfg.model.edge_decoding == 'concat':
            self.layer_post_mp = MLP(dim_in * 2, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)            
            # requires parameter
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        elif cfg.model.edge_decoding == 'summation':
            self.layer_post_mp = MLP(dim_in, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.add(v1, v2))
        else:
            if dim_out > 1:
                raise ValueError(
                    'Binary edge decoding ({})is used for multi-class '
                    'edge/link prediction.'.format(cfg.model.edge_decoding))
            self.layer_post_mp = MLP(dim_in, dim_in,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
    
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            # elif cfg.model.edge_decoding == 'cosine_similarity':
            #     self.decode_module = nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError('Unknown edge decoding {}.'.format(
                    cfg.model.edge_decoding))

    def _apply_index(self, batch):
        return batch.node_feature[batch.edge_label_index], \
               batch.edge_label

    # return node index for edges per batch
    def _apply_index_batch_pointwise(self, batch, sup_pos_edge_num, batch_idx, is_last_batch, batch_size=cfg.train.batch_size):
        if is_last_batch:
            # sup_pos_edge_num = int(batch.edge_label_index.shape[1])
            return batch.node_feature[batch.edge_label_index[:,batch_idx * batch_size:sup_pos_edge_num]], \
                batch.edge_label[batch_idx * batch_size:sup_pos_edge_num]
        else:
            return batch.node_feature[batch.edge_label_index \
            [:,batch_idx * batch_size:(batch_idx + 1) * batch_size]], \
            batch.edge_label[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat' and cfg.model.edge_decoding != 'summation':
            batch = self.layer_post_mp(batch)
        
        # if self.training:
        pred, label = torch.empty(0, 0).to(cfg.device), torch.empty(0, 0).to(cfg.device)

        batch_size = cfg.train.batch_size

        if cfg.dataset.load_type == 'pointwise':
            if self.training or cfg.model.eval_type != 'ranking':
                sup_pos_edge_num = int((batch.edge_label_index.shape[1]) // 2)
                batch_num = int(math.ceil(sup_pos_edge_num / batch_size))

                for i in range(batch_num):
                    if i == batch_num - 1:
                        pred_batch, label_batch = self._apply_index_batch_pointwise(batch, sup_pos_edge_num, i, True, batch_size)
                    else:
                        pred_batch, label_batch = self._apply_index_batch_pointwise(batch, sup_pos_edge_num, i, False, batch_size)
                    nodes_first = pred_batch[0]
                    nodes_second = pred_batch[1]
                    pred_batch = self.decode_module(nodes_first, nodes_second)
                    if i == 0:
                        pred = pred_batch
                        label = label_batch
                    else:
                        pred = torch.cat([pred, pred_batch], dim=0)
                        label = torch.cat([label, label_batch], dim=0)
        
                return pred, label

            else:
                embeddings = batch.node_feature
                num_users, num_items = batch.num_users.item(), batch.num_items.item()
                pred_matrix = np.zeros((num_users, num_items))
                for i in tqdm(range(num_users), desc='Evaluating'):
                    emb_u = embeddings[i, :]
                    emb_u = emb_u.repeat(num_items).view(num_items, -1)
                    emb_i = embeddings[np.arange(num_users, num_users + num_items), :]
                    pred = self.decode_module(emb_u, emb_i)
                    probs = pred.detach().cpu().numpy()
                    pred_matrix[i] = np.reshape(probs, [-1, ])
                return pred_matrix


class GNNGraphHead(nn.Module):
    '''Head of GNN, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''

    def __init__(self, dim_in, dim_out):
        super(GNNGraphHead, self).__init__()
        # todo: PostMP before or after global pooling
        self.layer_post_mp = MLP(dim_in, dim_out,
                                 num_layers=cfg.gnn.layers_post_mp, bias=True)
        self.pooling_fun = pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.graph_label

    def forward(self, batch):
        if cfg.dataset.transform == 'ego':
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch,
                                         batch.node_id_index)
        else:
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


# Head models for external interface
head_dict = {
    'node': GNNNodeHead,
    'edge': GNNEdgeHead,
    'link_pred': GNNEdgeHead,
    'graph': GNNGraphHead
}

head_dict = {**register.head_dict, **head_dict}
