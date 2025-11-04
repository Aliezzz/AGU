import os
import errno
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx
import scipy.sparse as sp

from tqdm import tqdm
from scipy.sparse import coo_matrix
from torch.nn import functional as F
from sklearn.metrics import f1_score
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph


def feature_reader(path):
    """
    Reading the sparse feature matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return features: Dense matrix of features.
    """
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = max(node_index) + 1
    feature_count = max(feature_index) + 1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    return features


def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"]).reshape(-1, 1)
    return target


def make_adjacency(graph, max_degree, sel=None):
    all_nodes = np.array(graph.nodes())

    # Initialize w/ links to a dummy node
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes + 1, max_degree)) + n_nodes).astype(int)

    if sel is not None:
        # only look at nodes in training set
        all_nodes = all_nodes[sel]

    for node in tqdm(all_nodes):
        neibs = np.array(list(graph.neighbors(node)))

        if sel is not None:
            neibs = neibs[sel[neibs]]

        if len(neibs) > 0:
            if len(neibs) > max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=False)
            elif len(neibs) < max_degree:
                extra = np.random.choice(neibs, max_degree - neibs.shape[0], replace=True)
                neibs = np.concatenate([neibs, extra])
            adj[node, :] = neibs

    return adj

def connected_component_subgraphs(graph):
    """
    Find all connected subgraphs in a networkx Graph

    Args:
        graph (Graph): A networkx Graph

    Yields:
        generator: A subgraph generator
    """
    for c in nx.connected_components(graph):
        yield graph.subgraph(c)


def check_exist(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def filter_edge_index(edge_index, node_indices, reindex=True):
    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index


def pyg_to_nx(data):
    """
    Convert a torch geometric Data to networkx Graph.

    Args:
        data (Data): A torch geometric Data.

    Returns:
        Graph: A networkx Graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(data.num_nodes))
    edge_index = data.edge_index.numpy()

    for u, v in np.transpose(edge_index):
        graph.add_edge(u, v)

    return graph


def edge_index_to_nx(edge_index, num_nodes):
    """
    Convert a torch geometric Data to networkx Graph by edge_index.
    Args:
        edge_index (Data.edge_index): A torch geometric Data.
        num_nodes (int): Number of nodes in a graph.
    Returns:
        Graph: networkx Graph
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(num_nodes))
    edge_index = edge_index.numpy()

    for u, v in np.transpose(edge_index):
        graph.add_edge(u, v)

    return graph


def filter_edge_index_1(data, node_indices):
    """
    Remove unnecessary edges from a torch geometric Data, only keep the edges between node_indices.
    Args:
        data (Data): A torch geometric Data.
        node_indices (list): A list of nodes to be deleted from data.

    Returns:
        data.edge_index: The new edge_index after removing the node_indices.
    """
    if isinstance(data.edge_index, torch.Tensor):
        data.edge_index = data.edge_index.cpu()

    edge_index = data.edge_index
    node_index = np.isin(edge_index, node_indices)

    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = data.edge_index[:, col_index]

    return np.searchsorted(node_indices, edge_index)


def random_split_dataset(n_samples):
    val_idx = np.random.choice(list(range(n_samples)), size=int(n_samples * 0.2), replace=False)
    remain = set(range(n_samples)) - set(val_idx)
    test_idx = np.random.choice(list(remain), size=int(n_samples * 0.2), replace=False)
    train_idx = list(remain - set(test_idx))

    return train_idx, val_idx, test_idx


def calc_f1(y_true, y_pred, mask, multilabel=False):
    if multilabel:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred_max = np.argmax(y_pred, axis=1)
    mask = mask.cpu()
    return f1_score(y_true[mask], y_pred_max[mask], average="micro")

def criterionKD(p, q, T=1.5):
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    soft_p = F.log_softmax(p / T, dim=1)
    soft_q = F.softmax(q / T, dim=1).detach()
    return loss_kl(soft_p, soft_q)


def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list[-1]


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj, r=0.5, n_nodes=None):
    if n_nodes is not None:
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)), shape=(n_nodes, n_nodes))
        adj.setdiag(0)
        adj = adj + sp.eye(n_nodes)
    else:
        adj.setdiag(0)
        adj = adj + sp.eye(adj.shape[0])

    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized

def transform_edge_index_to_adj(edge_index, n_nodes=None):
    adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(to_scipy_sparse_matrix(edge_index), n_nodes=n_nodes))
    return adj

def generate_neighbor_list(edge_index, num_nodes):
    neighbors = [[] for _ in range(num_nodes)]
    for (u, v) in edge_index.t().tolist():
        neighbors[u].append(v)
        neighbors[v].append(u)

    one_hop_neighbors = [None] * num_nodes
    two_hop_neighbors = [None] * num_nodes
    three_hop_neighbors = [None] * num_nodes

    for node in range(num_nodes):
        hop1 = set(neighbors[node])
        hop2_candidates = set()
        for n1 in hop1:
            hop2_candidates.update(neighbors[n1])

        hop2 = hop2_candidates - hop1 - {node}
        hop3_candidates = set()
        for n2 in hop2:
            hop3_candidates.update(neighbors[n2])

        hop3 = hop3_candidates - hop1 - hop2 - {node}

        one_hop_neighbors[node] = hop1
        two_hop_neighbors[node] = hop2
        three_hop_neighbors[node] = hop3

    for i in range(num_nodes):
        one_hop_neighbors[i] = list(one_hop_neighbors[i])
        two_hop_neighbors[i] = list(two_hop_neighbors[i])
        three_hop_neighbors[i] = list(three_hop_neighbors[i])
    
    return [one_hop_neighbors, two_hop_neighbors, three_hop_neighbors]

def generate_micro_features(node_feats, edge_index, num_layers, dataset_name):
    micro_features = []
    num_nodes = node_feats.shape[0]

    neighbor_list_file = os.path.join('lib_neighbors', '{}_neighbors.pkl'.format(dataset_name))
    os.makedirs(os.path.dirname(neighbor_list_file), exist_ok=True)

    if os.path.exists(neighbor_list_file):
        with open(neighbor_list_file, 'rb') as f:
            neighbor_list = pickle.load(f)
    else:
        neighbor_list = generate_neighbor_list(edge_index, num_nodes)
        with open(neighbor_list_file, 'wb') as f:
            pickle.dump(neighbor_list, f)

    one_hop_neighbors, two_hop_neighbors, three_hop_neighbors = neighbor_list

    print('Generating micro features...')
    for node in tqdm(np.arange(num_nodes)):
        if num_layers == 1:
            src, dst = one_hop_neighbors[node], two_hop_neighbors[node]
        elif num_layers == 2:
            src, dst = two_hop_neighbors[node], three_hop_neighbors[node]

        between_nodes = np.union1d(src, dst)
        if len(between_nodes) <= 1 or len(src) == 0 or len(dst) == 0:
            micro_features.append(node_feats[node])
        else:
            sub_edge_index, _ = subgraph(torch.tensor(between_nodes), edge_index, relabel_nodes=False)

            forward_edge_mask = np.isin(sub_edge_index[0], src) & np.isin(sub_edge_index[1], dst)
            between_edges = sub_edge_index[:, forward_edge_mask]

            if between_edges.shape[1] > 5:
                sel_edges = between_edges[:, torch.randint(between_edges.shape[1], (5,))]
            else:
                sel_edges = between_edges

            temp_features = []
            for sel_edge in sel_edges.T:
                delete_mask = ((edge_index[0] == sel_edge[0]) & (edge_index[1] == sel_edge[1])) | \
                            ((edge_index[0] == sel_edge[1]) & (edge_index[1] == sel_edge[0]))

                marginal_edge_index = edge_index[:, ~delete_mask]
                micro_adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(to_scipy_sparse_matrix(marginal_edge_index), n_nodes=num_nodes))
                micro_feature = propagate(node_feats, num_layers, micro_adj)
                temp_features.append(micro_feature[node])

            micro_features.append(torch.stack(temp_features, dim=0).mean(dim=0))

    micro_features = torch.stack(micro_features, dim=0)
    return micro_features

def filter_edge_index(edge_index, node_indices, reindex=True):
    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index
    

def get_dataset_train(data):

    train_indices = np.nonzero(data.train_mask.cpu().numpy())[0]
    test_indices = np.nonzero(data.test_mask.cpu().numpy())[0]
    edge_index = filter_edge_index(data.edge_index, train_indices, reindex=False)
    if edge_index.shape[1] == 0:
        edge_index = torch.tensor([[1, 2], [2, 1]])

    dataset_train = Data(x=data.x, edge_index=edge_index, y=data.y, train_mask=data.train_mask,
                         test_mask=data.test_mask, train_indices=train_indices, 
                         test_indices=test_indices)

    return dataset_train


def get_influence_nodes(args, unlearn_nodes, edge_index, hops=2):
    influenced_nodes = unlearn_nodes
    for _ in range(hops):
        target_nodes_location = np.isin(edge_index[0], influenced_nodes)
        neighbor_nodes = edge_index[1, target_nodes_location]
        influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
        influenced_nodes = np.unique(influenced_nodes)
    if args['unlearn_task'] == 'node':
        neighbor_nodes = np.setdiff1d(influenced_nodes, unlearn_nodes)
    else:
        neighbor_nodes = influenced_nodes
    return neighbor_nodes


def get_dataset_unlearn(args, data, unlearning_id, delete_edge_index=None):
    train_indices = np.nonzero(data.train_mask.cpu().numpy())[0]
    test_indices = np.nonzero(data.test_mask.cpu().numpy())[0]
    if args['unlearn_task'] == 'feature':
        x = data.x
        x[unlearning_id] = 0
        dataset_unlearn = Data(x=x, edge_index=data.edge_index, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, train_indices=train_indices, test_indices=test_indices)
    else:
        if args['unlearn_task'] == 'node':
            edge_index_unlearn = update_edge_index_unlearn(args, data.edge_index.cpu().numpy(), unlearning_id)
        elif args['unlearn_task'] == 'edge':
            edge_index_unlearn = update_edge_index_unlearn(args, data.edge_index.cpu().numpy(), unlearning_id, delete_edge_index)
        dataset_unlearn = Data(x=data.x, edge_index=edge_index_unlearn, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, train_indices=train_indices, test_indices=test_indices)

    return dataset_unlearn


def update_edge_index_unlearn(args, edge_index, delete_nodes, delete_edge_index=None):
    unique_indices = np.where(edge_index[0] < edge_index[1])[0]
    unique_edge_index = edge_index[:, unique_indices]

    if args['unlearn_task'] == 'edge':
        remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        remain_edges = edge_index[:, remain_indices]
    else:
        unique_edge_index = edge_index[:, unique_indices]
        delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                            np.isin(unique_edge_index[1], delete_nodes))
        remain_indices = np.logical_not(delete_edge_indices)
        remain_indices = np.where(remain_indices == True)[0]
        remain_edges = unique_edge_index[:, remain_indices]
    remain_edges = to_undirected(torch.from_numpy(remain_edges))

    return remain_edges