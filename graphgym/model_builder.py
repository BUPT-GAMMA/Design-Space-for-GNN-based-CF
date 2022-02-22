import torch

from graphgym.config import cfg
from graphgym.models.gnn import GNN, MCGNN

from graphgym.contrib.network import *
import graphgym.register as register

network_dict = {
    'gnn': GNN,
    'mcgnn': MCGNN,
}
network_dict = {**register.network_dict, **network_dict}


def create_model(datasets=None, to_device=True, dim_in=None, dim_out=None):
    dim_in = datasets[0].num_node_features if dim_in is None else dim_in
    # add one-hot encoder without node features
    if dim_in == 0:
        dim_in = datasets[0].num_nodes[0]
    dim_out = 1 if dim_out is None else dim_out
    # dim_out = datasets[0].num_labels if dim_out is None else dim_out
    # # binary classification, output dim = 1
    # if 'classification' in cfg.dataset.task_type and dim_out == 2:
    #     dim_out = 1

    model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)

    if to_device:
        model.to(torch.device(cfg.device))
    return model
