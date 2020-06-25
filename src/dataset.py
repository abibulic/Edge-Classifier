import copy
import numpy as np

import torch
from torch.utils.data import Dataset

from graph_network import utils_torch

from qm9_dataloader.utils import qm9_nodes, qm9_edges
from qm9_dataloader.graph_reader import xyz_graph_reader 

class Qm9(Dataset):

    def __init__(self, paths, args):
        self.paths = paths
        self.args = args
        
    def __getitem__(self, index):
        g, target = xyz_graph_reader(self.paths[index])
        if self.args.nodes_transform:
            h = qm9_nodes(g)

        if self.args.edge_transform:
            g, e = qm9_edges(g, self.args.e_representation)

        # if self.args.target_transform:
        #     target = self.target_transform(target)

        return self.create_graph_dictionary(h, e), target

    def __len__(self):
        return len(self.paths)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform
    
    def create_graph_dictionary(self, h, e):
        sender_params = []
        receiver_params = []
        edges = []
        for sr, edg in e.items():
            edges.append(edg)
            sender_params.append(sr[0])
            receiver_params.append(sr[1])
        graph_dicts ={"globals": 0,
                    "nodes": np.float32(h),
                    "edges": np.float32(edges),
                    "senders": np.int32(sender_params),
                    "receivers": np.int32(receiver_params)}
        return graph_dicts

    @staticmethod
    def collate_fn(batch):
        data = []
        target = []
        for item in batch:
            temp = copy.deepcopy(item[0])
            target.append(temp['edges'][:, 1:])
            temp['edges'] = np.expand_dims(temp['edges'][:,0], axis=1)
            data.append(temp)
            del temp

        #data = [item[0] for item in batch]        
        #target = [item[1] for item in batch]
        data = utils_torch.data_dicts_to_graphs_tuple(data)
        #target = torch.LongTensor(target)
        target = torch.FloatTensor(np.concatenate(target, axis=0))
        return data, target