import logging
import pickle
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import numpy as np
from exp import Exp
from lib_gnn_model.node_classifier import NodeClassifier
from torch_geometric.utils import to_undirected
from lib_utils.utils import calc_f1, criterionKD, transform_edge_index_to_adj, propagate, generate_micro_features, generate_neighbor_list, get_dataset_train

class ExpAGU(Exp):
    def __init__(self, args):
        super(ExpAGU, self).__init__(args)
        self.logger = logging.getLogger('ExpAGU')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = self.data_store.load_raw_data()
        self.num_feats = self.data.num_features
        self.num_layers = self.args['n_layers']
        self.target_model_name = self.args['target_model']

        self.deleted_nodes = np.array([])     
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])
        self.data.margin_nodes = np.array([])

        self.train_test_split()
        self.unlearning_request()
        self.determine_target_model()

        self.adj = transform_edge_index_to_adj(self.data.edge_index)
        
        self.target_model.data = self.data
        if self.target_model_name in ['GCN', 'SGC']:
            self.marigin_filter()

        train_f1s = np.empty((0))
        train_times = np.empty((0))

        unlearn_f1s = np.empty((0))
        unlearn_times = np.empty((0))

        for _ in range(self.args['num_runs']):
            train_time, _ = self._train_model()
            f1_score = self.evaluate()

            train_f1s = np.append(train_f1s, f1_score)
            train_times = np.append(train_times, train_time)

            unlearn_time, unlearn_f1 = self.agu_training()
            unlearn_f1s = np.append(unlearn_f1s, unlearn_f1)
            unlearn_times = np.append(unlearn_times, unlearn_time)

        self.logger.info('Unlearn F1: %.4f, Unlearn Time: %.4f' % (np.mean(unlearn_f1s), np.mean(unlearn_times)))

    def train_test_split(self):
        if self.args['is_split']:
            self.train_indices, self.test_indices = train_test_split(np.arange(self.data.num_nodes),
                                                                     test_size=self.args['test_ratio'],
                                                                     random_state=100)
        
            self.data_store.save_train_test_split(self.train_indices, self.test_indices)
            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split()
            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))

    def unlearning_request(self):
        self.dataset_train = get_dataset_train(self.data)

        self.data.x_unlearn = self.data.x.clone()
        self.data.retrain_mask = self.data.train_mask.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.numpy()
        unique_indices = np.where(
            (edge_index[0] < edge_index[1]) &
            (~np.isin(edge_index[0], self.test_indices)) &
            (~np.isin(edge_index[1], self.test_indices)))[0]

        if self.args["unlearn_task"] == 'node':
            unique_nodes = np.random.choice(self.train_indices,
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            self.data.node_removed = unique_nodes
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)
            self.data.retrain_mask[unique_nodes] = False

        if self.args["unlearn_task"] == 'edge':
            remove_indices = np.random.choice(
                unique_indices,
                int(unique_indices.shape[0] * self.args['unlearn_ratio']),
                replace=False)
            remove_edges = edge_index[:, remove_indices]
            unique_nodes = np.unique(remove_edges)
            remove_mask = torch.zeros(self.data.edge_index.shape[1], dtype=torch.bool)
            remove_mask[remove_indices] = True
            self.data.edge_index_removed = remove_edges
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_indices)
            
        if self.args["unlearn_task"] == 'feature':
            unique_nodes = np.random.choice(self.train_indices,
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            self.data.x_unlearn[unique_nodes] = 0.
            self.feature_nodes = unique_nodes

        if self.args["unlearn_task"] in ['node', 'edge']:
            self.find_potentials()

        self.find_k_hops(unique_nodes)
        self.data.core_node = unique_nodes
        self.data.retain_node = np.setdiff1d(self.train_indices, unique_nodes)

    def find_potentials(self):
        edge_index_contrast = []
        edge_index_potential = []

        neighbor_list_file = os.path.join('lib_neighbors', '{}_neighbors.pkl'.format(self.args['dataset_name']))
        os.makedirs(os.path.dirname(neighbor_list_file), exist_ok=True)

        if os.path.exists(neighbor_list_file):
            with open(neighbor_list_file, 'rb') as f:
                neighbor_list = pickle.load(f)
        else:
            neighbor_list = generate_neighbor_list(self.data.edge_index, self.data.num_nodes)
            with open(neighbor_list_file, 'wb') as f:
                pickle.dump(neighbor_list, f)

        one_hop_neighbors, two_hop_neighbors, three_hop_neighbors = neighbor_list

        if self.args["unlearn_task"] == 'edge':
            edge_index_removed = self.data.edge_index_removed
            for src, dst in zip(edge_index_removed[0], edge_index_removed[1]):
                if self.args["nei_range"] == 3:
                    src_indirect_neighbors = np.union1d(two_hop_neighbors[src], three_hop_neighbors[src])
                    dst_indirect_neighbors = np.union1d(two_hop_neighbors[dst], three_hop_neighbors[dst])
                elif self.args["nei_range"] == 2:
                    src_indirect_neighbors = two_hop_neighbors[src]
                    dst_indirect_neighbors = two_hop_neighbors[dst]
                common_neighbors = np.intersect1d(src_indirect_neighbors, dst_indirect_neighbors)
                common_neighbors = common_neighbors[~np.isin(common_neighbors, [src, dst])]
                if len(common_neighbors) >= 2:
                    edge_index_contrast.append([src, dst])
                    selected_neighbors = np.random.choice(common_neighbors, size=2, replace=False)
                    edge_index_potential.append([selected_neighbors[0], selected_neighbors[1]])

        if self.args["unlearn_task"] == 'node':
            node_removed = self.data.node_removed
            for node in node_removed:
                neighbors = np.random.choice(one_hop_neighbors[node], 
                                                size=min(self.args["n_nei_select"], len(one_hop_neighbors[node])), 
                                                replace=False)
                if len(neighbors) > 0:
                    for neighbor in neighbors:
                        src_indirect_neighbors = np.array(two_hop_neighbors[node])
                        dst_indirect_neighbors = np.array(two_hop_neighbors[neighbor])
                        common_neighbors = np.intersect1d(src_indirect_neighbors, dst_indirect_neighbors)
                        common_neighbors = common_neighbors[~np.isin(common_neighbors, [node, neighbor])]
                        if len(common_neighbors) >= 2:
                            edge_index_contrast.append([node, neighbor])
                            selected_neighbors = np.random.choice(common_neighbors, size=2, replace=False)
                            edge_index_potential.append([selected_neighbors[0], selected_neighbors[1]])

        self.data.edge_index_contrast = torch.from_numpy(np.array(edge_index_contrast).T).long()
        self.data.edge_index_potential = torch.from_numpy(np.array(edge_index_potential).T).long()

    def find_k_hops(self, unique_nodes):
        edge_index = self.data.edge_index.numpy()
        
        if self.args["unlearn_task"] == 'node':
            hops = self.num_layers + 1
        else:
            hops = self.num_layers

        def compute_influenced_nodes(unique_nodes, edge_index, hops):
            influenced_nodes = unique_nodes
            for _ in range(hops):
                target_nodes_location = np.isin(edge_index[0], influenced_nodes)
                neighbor_nodes = edge_index[1, target_nodes_location]
                influenced_nodes = np.unique(np.append(influenced_nodes, neighbor_nodes))
            neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
            return neighbor_nodes
        
        self.data.ori_affected_nodes = compute_influenced_nodes(unique_nodes, edge_index, hops)
        self.data.true_affected_nodes = compute_influenced_nodes(unique_nodes, edge_index, hops - 1)
        self.data.marginal_nodes = np.setdiff1d(self.data.ori_affected_nodes, self.data.true_affected_nodes)

        if self.args["unlearn_task"] == 'feature' or 'partial_feature':
            self.feature_nodes = unique_nodes
            self.influence_nodes = self.data.true_affected_nodes
        if self.args["unlearn_task"] == 'node':
            self.deleted_nodes = unique_nodes
            self.influence_nodes = self.data.ori_affected_nodes
        if self.args["unlearn_task"] == 'edge':
            self.influence_nodes = np.union1d(unique_nodes, self.data.ori_affected_nodes)

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_edge_index = edge_index[:, unique_indices]

        if self.args["unlearn_task"] == 'edge':
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

    def determine_target_model(self):
        num_classes = self.data.num_classes
        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args)

    def evaluate(self):
        self.target_model.model.eval()
        _, out = self.target_model.model(self.data.x, self.data.edge_index)
        y = self.data.y.cpu()
        y_hat = F.log_softmax(out, dim=1).cpu().detach().numpy()
        test_f1 = calc_f1(y, y_hat, self.data.test_mask)
        return test_f1

    def _train_model(self):
        start_time = time.time()
        res = self.target_model.train_model((self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        train_time = time.time() - start_time

        return train_time, res
    
    def marigin_filter(self):
        temp_features = self.data.x.clone()
        marginal_nodes = self.data.marginal_nodes
        ori_adj = transform_edge_index_to_adj(self.data.edge_index, self.data.num_nodes)
        ori_features = propagate(temp_features, self.num_layers, ori_adj)

        unlearn_adj = transform_edge_index_to_adj(self.data.edge_index_unlearn, self.data.num_nodes)
        unlearn_features = propagate(temp_features, self.num_layers, unlearn_adj)

        micro_feat_file = os.path.join('lib_micro_feats', '{}_{}_hop_micro_feat.pt'.format(self.args['dataset_name'],
                                                                                             self.num_layers))
        os.makedirs(os.path.dirname(micro_feat_file), exist_ok=True)
        if not os.path.exists(micro_feat_file):
            micro_features = generate_micro_features(self.data.x, self.data.edge_index, self.num_layers, self.args['dataset_name'])
            torch.save(micro_features, micro_feat_file)
        else:
            micro_features = torch.load(micro_feat_file)

        cos = nn.CosineSimilarity()
        ori_unlearn_cos = cos(ori_features[marginal_nodes], unlearn_features[marginal_nodes])
        ori_micro_cos = cos(ori_features[marginal_nodes], micro_features[marginal_nodes])

        self.data.margin_nodes = marginal_nodes[torch.abs(ori_unlearn_cos - ori_micro_cos) > self.args['margin_threshold']]

    def agu_training(self):
        island_nodes = []

        if self.args["unlearn_task"] == 'feature':
            affected_nodes = self.data.true_affected_nodes 
        else:
            if self.target_model_name in ['GCN', 'SGC']:
                    affected_nodes = np.append(self.data.true_affected_nodes, self.data.margin_nodes)
            else:
                affected_nodes = self.data.true_affected_nodes

        with torch.no_grad():
            self.target_model.model.eval()
            _, ori_x = self.target_model.model(self.data.x, self.data.edge_index)
            _, unlearn_x = self.target_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)

            if self.args["unlearn_task"] == 'edge':
                affected_nodes = np.append(affected_nodes, self.data.core_node)

            cos = nn.CosineSimilarity()
            total_cos = cos(ori_x[affected_nodes], unlearn_x[affected_nodes])
            k = int(len(affected_nodes) * self.args['affected_ratio'])
            affected_nodes = affected_nodes[torch.topk(total_cos, k, largest=False).indices.cpu().numpy()]

        if self.args["unlearn_task"] == 'node':
            island_nodes = self.data.core_node
        elif self.args["unlearn_task"] == 'edge':
            island_nodes = np.setdiff1d(
                self.data.core_node,
                self.data.edge_index_unlearn.unique().cpu().numpy())
        elif self.args["unlearn_task"] == 'feature':
            island_nodes = self.data.core_node

        empty_edges = torch.from_numpy(np.array([[], []], dtype=np.int64)).to(self.device)
        self.agu_n_margin = len(self.data.margin_nodes)
        self.agu_n_affected = len(affected_nodes)
        self.agu_n_island = len(island_nodes)

        with torch.no_grad():
            self.target_model.model.eval()
            _, preds_logits = self.target_model.model(self.data.x, self.data.edge_index)
            _, empty_preds_logits = self.target_model.model(self.data.x, empty_edges)

            preds = torch.argmax(preds_logits, axis=1).type_as(self.data.y)

        edge_loss_fcn = nn.MSELoss(reduction='mean')
        temp_model = NodeClassifier(self.num_feats, self.data.num_classes, self.args)
        temp_model.model.load_state_dict(self.target_model.model.state_dict())
        optimizer = torch.optim.SGD(temp_model.model.parameters(), lr=self.args['agu_unlearn_lr'])

        start_time = time.time()
        for _ in range(self.args['agu_epochs']):
            temp_model.model.train()
            optimizer.zero_grad()

            loss_edge, loss_island = 0, 0
            _, unlearn_out = temp_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)

            if self.args['unlearn_task'] in ['node', 'edge'] and self.data.edge_index_contrast[0].size(0) > 0:
                deleted_feats = torch.cat([unlearn_out[self.data.edge_index_contrast[0]], unlearn_out[self.data.edge_index_contrast[1]]], dim=1)
                potential_feats = torch.cat([preds_logits[self.data.edge_index_potential[0]], preds_logits[self.data.edge_index_potential[1]]], dim=1)
                loss_edge = edge_loss_fcn(potential_feats, deleted_feats)
            
            if self.args['unlearn_task'] in ['node', 'feature']:
                loss_island = criterionKD(unlearn_out[island_nodes], empty_preds_logits[island_nodes])

            loss_affcted = F.cross_entropy(unlearn_out[affected_nodes], preds[affected_nodes])
            loss = -loss_island + loss_affcted + self.args['edge_weight'] * loss_edge

            loss.backward()
            optimizer.step()
        unlearn_time = time.time() - start_time

        temp_model.model.eval()
        _, test_out = temp_model.model(self.data.x_unlearn, self.data.edge_index_unlearn)

        out = F.softmax(test_out, dim=-1)

        y_hat = out.cpu().detach().numpy()
        y = self.data.y.cpu()
        test_f1 = calc_f1(y, y_hat, self.data.test_mask)

        return unlearn_time, test_f1