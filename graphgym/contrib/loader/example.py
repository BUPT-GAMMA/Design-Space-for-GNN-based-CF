from graphgym.train import train
from networkx.readwrite import graph6
from numpy.lib.arraysetops import unique
import pandas as pd
import numpy as np
import torch
import scipy.io
import scipy.sparse as sp
import networkx as nx
import h5py
import pickle as pkl

from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from torch_geometric.datasets import *
from torch_geometric.data import Data

from graphgym.register import register_loader
from graphgym.config import cfg

dtypes = {
        'user': np.int64, 'item': np.int64,
        'rating': np.float32, 'timestamp': float}

def load_dataset_example(format, name, dataset_dir):
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name == 'QM7b':
            dataset_raw = QM7b(dataset_dir)
            graphs = GraphDataset.pyg_to_graphs(dataset_raw)
            return graphs

def load_dataset_ml_100k(format, name, dataset_dir):
    dataset_dir = f'{dataset_dir}/{name}'
    if 'ml-100k' in name:
        if cfg.dataset.load_type == 'pointwise':
            dataset_raw = pd.read_csv(f'{dataset_dir}/u.data', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])

            train_data = pd.read_csv(f'{dataset_dir}/ua.base', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
            test_data = pd.read_csv(f'{dataset_dir}/ua.test', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])

            train_array = train_data.values.tolist()
            train_array = np.array(train_array)
            test_array = test_data.values.tolist()
            test_array = np.array(test_array)
            
            data_array = np.concatenate([train_array, test_array], axis=0)

            users = data_array[:, 0].astype(dtypes['user'])
            items = data_array[:, 1].astype(dtypes['item'])
            ratings = data_array[:, 2].astype(dtypes['rating'])

            # train_mean = np.mean(train_array[:, 2])
            # train_std = np.std(train_array[:, 2])
            # ratings = (ratings - train_mean) / train_std
            
            users, items = map_node_label(users, items)
            num_users = int(users.max() + 1)
            num_items = int(items.max() - users.max())
            num_nodes = num_users + num_items
            
            num_train = train_array.shape[0]
            num_test = test_array.shape[0]
            num_val = int(np.ceil(num_train * 0.1))
            num_train -= num_val

            rated_pairs = [(u, i, float(r)) for u, i, r in zip(users, items, ratings)]

            train_edges, val_edges, test_edges = split(rated_pairs, num_train, num_val, num_test)

            graph = gen_rated_graph(num_users, num_items, rated_pairs, train_edges, val_edges, test_edges)

            return [graph]
           


def load_dataset_ml_1m(format, name, dataset_dir):
    dataset_dir = f'{dataset_dir}/{name}'
    if 'ml-1m' in name:
        if cfg.dataset.load_type == 'pointwise':
            dataset_raw = pd.read_csv(f'{dataset_dir}/ratings.dat', sep='::', header=None, engine='python', names=['user', 'item', 'rating', 'timestamp'])
            
            users = np.array(dataset_raw['user'])
            items = np.array(dataset_raw['item'])
            ratings = np.array(dataset_raw['rating'])

            users, items = map_node_label(users, items)
            num_users = int(users.max() + 1)
            num_items = int(items.max() - users.max())

            rated_pairs = [(u, i, float(r)) for u, i, r in zip(users, items, ratings)]
            num_ratings = len(rated_pairs)

            num_train = int(np.ceil(num_ratings * cfg.dataset.split[0]))
            num_val = int(np.ceil(num_ratings * cfg.dataset.split[1]))
            num_test = num_ratings - num_train - num_val

            np.random.seed(cfg.seed)
            np.random.shuffle(rated_pairs)

            train_edges, val_edges, test_edges = split(rated_pairs, num_train, num_val, num_test)

            graph = graph = gen_rated_graph(num_users, num_items, rated_pairs, train_edges, val_edges, test_edges)

            return [graph]

def load_dataset_monti(format, name, dataset_dir):
    if name in ['douban', 'flixster', 'yahoo-music']:
        if cfg.dataset.load_type == 'pointwise':
            dataset_dir = f'{dataset_dir}/{name}/training_test_dataset.mat'

            M = load_matlab_file(dataset_dir, 'M')
            Otraining = load_matlab_file(dataset_dir, 'Otraining')
            Otest = load_matlab_file(dataset_dir, 'Otest')

            train_user, train_item = np.where(Otraining)[0].astype(dtypes['user']), np.where(Otraining)[1].astype(dtypes['item'])
            test_user, test_item = np.where(Otest)[0].astype(dtypes['user']), np.where(Otest)[1].astype(dtypes['item'])

            users = np.concatenate([train_user, test_user], axis=0)
            items = np.concatenate([train_item, test_item], axis=0)
            ratings = M[users, items]

            # Convert the rating scale of flixster to [1, 10]
            if name == 'flixster':
                ratings = ratings * 2

            users, items = map_node_label(users, items)
            num_users = int(users.max() + 1)
            num_items = int(items.max() - users.max())
            num_nodes = int(items.max() + 1)

            num_train = train_user.shape[0]
            num_test = test_user.shape[0]
            num_val = int(np.ceil(num_train * 0.2))
            num_train -= num_val

            rated_pairs = [(u, i, float(r)) for u, i, r in zip(users, items, ratings)]

            train_edges, val_edges, test_edges = split(rated_pairs, num_train, num_val, num_test)

            graph = gen_rated_graph(num_users, num_items, rated_pairs, train_edges, val_edges, test_edges)

            return [graph]


def load_dataset_amazon(formast, name, dataset_dir):
    if 'amazon' in name or 'yelp2020' in name:
        if cfg.dataset.load_type == 'pointwise':
            dataset_file = f'{dataset_dir}/{name}/filtered_rating.csv'
            
            dataset_raw = pd.read_csv(dataset_file, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')

            users = np.array(dataset_raw['user'])
            items = np.array(dataset_raw['item'])
            ratings = np.array(dataset_raw['rating'])

            users, items = map_node_label(users, items)
            num_users = int(users.max() + 1)
            num_items = int(items.max() - users.max())

            rated_pairs = [(u, i, float(r)) for u, i, r in zip(users, items, ratings)]
            num_ratings = len(rated_pairs)

            num_train = int(np.ceil(num_ratings * cfg.dataset.split[0]))
            num_val = int(np.ceil(num_ratings * cfg.dataset.split[1]))
            num_test = num_ratings - num_train - num_val

            np.random.seed(cfg.seed)
            np.random.shuffle(rated_pairs)

            train_edges, val_edges, test_edges = split(rated_pairs, num_train, num_val, num_test)

            graph = gen_rated_graph(num_users, num_items, rated_pairs, train_edges, val_edges, test_edges)

            return [graph]

def load_dataset_epinions(format, name, dataset_dir):
    if name in ['epinions']:
        if cfg.dataset.load_type == 'pointwise':
            dataset_file = f'{dataset_dir}/{name}/ratings_data.txt'

            dataset_raw = pd.read_csv(dataset_file, sep=' ', names=['user', 'item', 'rating'], engine='python')

            users = np.array(dataset_raw['user'])
            items = np.array(dataset_raw['item'])
            ratings = np.array(dataset_raw['rating'])

            users, items = map_node_label(users, items)
            num_users = int(users.max() + 1)
            num_items = int(items.max() - users.max())

            rated_pairs = [(u, i, float(r)) for u, i, r in zip(users, items, ratings)]
            num_ratings = len(rated_pairs)

            num_train = int(np.ceil(num_ratings * cfg.dataset.split[0]))
            num_val = int(np.ceil(num_ratings * cfg.dataset.split[1]))
            num_test = num_ratings - num_train - num_val

            np.random.seed(cfg.seed)
            np.random.shuffle(rated_pairs)

            train_edges, val_edges, test_edges = split(rated_pairs, num_train, num_val, num_test)

            graph = gen_rated_graph(num_users, num_items, rated_pairs, train_edges, val_edges, test_edges)

            return [graph]



register_loader('example', load_dataset_example)
register_loader('ml-100k', load_dataset_ml_100k)
register_loader('ml-1m', load_dataset_ml_1m)
register_loader('monti', load_dataset_monti)
register_loader('amazon', load_dataset_amazon)
register_loader('epinions', load_dataset_epinions)


def map_node_label(user, item):
    """
    mark the node label starting from 0 and filter user/item label without interaction
    """
    unique_user = np.unique(user)
    unique_item = np.unique(item)
    user_label_map = {}
    for l, u in enumerate(unique_user):
        user_label_map[u] = l
    item_label_map = {}
    for l, i in enumerate(unique_item):
        item_label_map[i] = l

    mapped_user = []
    mapped_item = []
    mapped_user = [user_label_map[u] for u in user]
    mapped_item = [item_label_map[i] for i in item]
    mapped_user = np.asarray(mapped_user, dtype=user.dtype)
    mapped_item = np.asarray(mapped_item, dtype=item.dtype)

    mapped_item += mapped_user.max() + 1
    return mapped_user, mapped_item

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    raw_file = h5py.File(path_file, 'r')
    raw_data = raw_file[name_field]
    try:
        if 'ir' in raw_data.keys():
            data = np.asarray(raw_data['data'])
            ir = np.asarray(raw_data['ir'])
            jc = np.asarray(raw_data['jc'])
            output = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        output = np.asarray(raw_data).astype(np.float32).T

    raw_file.close()

    return output


def split(pairs, num_train, num_val, num_test):
    """
    split users/items according to the given ratio
    """
    # pairs = [(u, i) for u, i in zip(users, items)]
    train_pairs = pairs[:num_train + num_val]
    test_pairs = pairs[num_train + num_val:num_train + num_val + num_test]

    ## split val set from train set
    np.random.seed(cfg.seed)
    np.random.shuffle(train_pairs)
    val_edges = train_pairs[0:num_val]
    train_edges = train_pairs[num_val:num_val + num_train]
    test_edges = test_pairs

    if len(pairs[0]) == 3:
        train_edges = [(u_i_r[0], u_i_r[1]) for u_i_r in train_edges]
        val_edges = [(u_i_r[0], u_i_r[1]) for u_i_r in val_edges]
        test_edges = [(u_i_r[0], u_i_r[1]) for u_i_r in test_edges]

    return train_edges, val_edges, test_edges


def gen_graph(num_nodes, pairs, train_edges, val_edges, test_edges):
    """
    generate the deepsnap.Graph given the custom split
    """
    # num_nodes = int(pairs[1].max() + 1)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    # G.add_nodes_from(range(num_users), node_type='user')
    # G.add_nodes_from(range(num_users, num_users + num_items), node_type='item')

    # node_feature = torch.eye(num_nodes)

    # for i in G.nodes:
    #     G.nodes[i]['node_feature'] = node_feature[i] 
    G.add_edges_from(pairs)
    dataset_custom = {}
    dataset_custom['general_splits'] = []
    dataset_custom['general_splits'].append(train_edges)
    dataset_custom['general_splits'].append(val_edges)
    dataset_custom['general_splits'].append(test_edges)
    dataset_custom['task'] = 'link_pred'
    graph = Graph(G, custom=dataset_custom)

    return graph


def gen_rated_graph(num_users, num_items, rated_pairs, train_edges=None, val_edges=None, test_edges=None):
    """
    generate the deepsnap.Graph with rating info as the edge weight, and split the graph if the split scheme is offered
    """

    G = nx.Graph()
    G.add_nodes_from(range(num_users), node_type='user')
    G.add_nodes_from(range(num_users, num_users + num_items), node_type='item')

    G.add_weighted_edges_from(rated_pairs, weight='edge_label')

    if train_edges is None:
        graph = Graph(G,  num_users=num_users, num_items=num_items)
    else:
        dataset_custom = {}
        dataset_custom['task'] = 'link_pred'
        dataset_custom['negative_edges'] = [[(0,0)], [(0, 0)], [(0, 0)]]
        dataset_custom['general_splits'] = []
        dataset_custom['general_splits'].append(train_edges)
        dataset_custom['general_splits'].append(val_edges)
        dataset_custom['general_splits'].append(test_edges)
        # rating prediction doesn't need valid negative sampling
        # dataset_custom['negative_edges'] = [[torch.tensor([0]), torch.tensor([0])]]
        graph = Graph(G, custom=dataset_custom, num_users=num_users, num_items=num_items)

    return graph



