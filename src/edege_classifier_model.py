import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from dataset import Qm9
from graph_network import modules, graphs, utils_torch
from graph_network.blocks import broadcast_receiver_nodes_to_edges, broadcast_sender_nodes_to_edges

def list_files(root_dir, ext='.xyz'):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file).replace("\\","/"))
    return file_list


class CollectNeighboursAndEdgesToNodes(torch.nn.Module):
    def __init__(self):
        super(CollectNeighboursAndEdgesToNodes, self).__init__()
    def forward(self, graph):
        neighbours_final = []
        edges_final = []
        for n in range(graph.nodes.shape[0]):
            neighbours = []
            edges = []
            idx_n = graph.receivers[(graph.senders==n).nonzero().view(-1).tolist()].tolist() + graph.senders[(graph.receivers==n).nonzero().view(-1).tolist()].tolist()
            idx_e = (graph.senders==n).nonzero().view(-1).tolist() +(graph.receivers==n).nonzero().view(-1).tolist()
            for i in range(4): # 4 je max broj susjeda i veza
                if(i < len(idx_n)):
                    neighbours.append(graph.nodes[idx_n[i]])
                else:
                    neighbours.append(torch.zeros(graph.nodes[0].shape).to(torch.device('cuda')))
                if(i < len(idx_e)):
                    edges.append(graph.edges[idx_e[i]])
                else:
                    edges.append(torch.zeros(graph.edges[0].shape).to(torch.device('cuda')))

            neighbours_final.append(torch.cat(neighbours, axis=-1))
            edges_final.append(torch.cat(edges, axis=-1))
        return torch.cat(neighbours_final, axis=-1).view(graph.nodes.shape[0], -1), torch.cat(edges_final, axis=-1).view(graph.nodes.shape[0], -1)


class Permutator(nn.Module):
    def __init__(self, inpit_size):
        super(Permutator, self).__init__()

        self.model = torch.nn.Sequential(
                    torch.nn.Linear(inpit_size, 64),
                    #torch.nn.Dropout(),
                    torch.nn.SELU(),
                    torch.nn.Linear(64, 64),
                    #torch.nn.Dropout(),
                    torch.nn.SELU(),
                    torch.nn.Linear(64, inpit_size),
                    )

    def forward(self, data):
        return self.model(data)


class EdgeClassifier(pl.LightningModule):
    def __init__(self, args):
        super(EdgeClassifier, self).__init__()

        self.hparams = args
        
        self.permutate_nodes = Permutator(26)
        
        self.update_edges = torch.nn.Sequential(
                                torch.nn.Linear(27, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 27),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(27, 1),
                                )

        self.edges_aggregator = CollectNeighboursAndEdgesToNodes()

        self.permutate_neighbours = Permutator(52)

        self.permutate_edges = Permutator(4)
        
        self.update_nodes = torch.nn.Sequential(
                                torch.nn.Linear(69, 50),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(50, 25),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(25, 13),
                                )

        self.predict = torch.nn.Sequential(
                                torch.nn.Linear(27, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 27),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(27, 4),
                                )
        
        self.sigmoid = torch.nn.Sigmoid()

        #losses
        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()

    #TODO: dati intuitivnije ime funkciji
    def pred_real_sum(self, graph, pred, target, node_or_edge):
        pred_sum = []
        real_sum = []
        if(node_or_edge=='edge'):
            index_cumsum = torch.cumsum(graph.n_edge, dim=0)
        else:
            index_cumsum = torch.cumsum(graph.n_node, dim=0)
        remember_sum_pred = 0
        remember_sum_real = 0
        for idx in index_cumsum:
            pred_sum.append(torch.sum(pred[:idx])-remember_sum_pred)
            remember_sum_pred = torch.sum(pred[:idx])

            real_sum.append(torch.sum(target[:idx])-remember_sum_real)
            remember_sum_real = torch.sum(target[:idx])

        pred_sum = torch.stack(pred_sum)
        real_sum = torch.stack(real_sum)
        return pred_sum, real_sum


    def forward(self, graph, optimizer_idx):
        # 2 noda za svaki edge
        collect_nodes_for_edges = []
        collect_nodes_for_edges.append(broadcast_receiver_nodes_to_edges(graph))
        collect_nodes_for_edges.append(broadcast_sender_nodes_to_edges(graph))
        collected_pair_nodes = torch.cat(collect_nodes_for_edges, axis=-1)

        h1_pair_nodes = self.permutate_nodes(collected_pair_nodes)

        if optimizer_idx == 0:
            return self.pred_real_sum(graph, h1_pair_nodes, collected_pair_nodes, 'edge')


        h1_edges = graph.edges
        for _ in range(3):
            temp = torch.cat([h1_edges, h1_pair_nodes], axis=-1)
            h1_edges = self.update_edges(temp)
        
        neighbours_of_nodes, edges_of_nodes = self.edges_aggregator(graph.replace(edges=h1_edges))
        
        h_neighbours = self.permutate_neighbours(neighbours_of_nodes)
        h2_edges = self.permutate_edges(edges_of_nodes)

        if optimizer_idx == 1:
            return self.pred_real_sum(graph, h_neighbours, neighbours_of_nodes, 'node')
        
        if optimizer_idx == 2:
            return self.pred_real_sum(graph, h2_edges, edges_of_nodes, 'node')

        h2_nodes = graph.nodes
        for _ in range(3):
            temp2 = torch.cat([h2_nodes, h_neighbours, h2_edges], axis=-1)
            h2_nodes = self.update_nodes(temp2)

        collect_nodes_for_edges2 = []
        collect_nodes_for_edges2.append(broadcast_receiver_nodes_to_edges(graph.replace(nodes=h2_nodes)))
        collect_nodes_for_edges2.append(broadcast_sender_nodes_to_edges(graph.replace(nodes=h2_nodes)))
        collected_nodes2 = torch.cat(collect_nodes_for_edges2, axis=-1)

        out = self.predict(torch.cat([h1_edges, collected_nodes2], axis=-1))

        return self.sigmoid(out)
        
    def prepare_data(self):
        files = list_files(self.hparams.data_path, ext='.xyz')
        train_data, valid_data = train_test_split(files, test_size=self.hparams.valid_split, random_state=0, shuffle=True)
        self.data_train = Qm9(train_data, self.hparams)
        self.data_valid = Qm9(valid_data, self.hparams)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.data_train,
                                               batch_size=self.hparams.batch_size, shuffle=True,
                                               collate_fn=Qm9.collate_fn,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.data_valid,
                                               batch_size=1, shuffle=False,
                                               collate_fn=Qm9.collate_fn,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return valid_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.data_valid,
                                               batch_size=1, shuffle=False,
                                               collate_fn=Qm9.collate_fn,
                                               num_workers=self.hparams.num_workers, pin_memory=False)
        return test_loader

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, verbose=True, patience=30)

        optimizer2 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, verbose=True, patience=30)

        optimizer3 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, verbose=True, patience=30)

        optimizer4 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler4 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, verbose=True, patience=30)
        
        return [optimizer1, optimizer2, optimizer3, optimizer4], [scheduler1, scheduler2, scheduler3, scheduler4]
    
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    #     # update optimizer every 8 steps (virtually increase of batch size)
    #     if optimizer_i == 0:
    #         if batch_nb % 8 == 0 :
    #             optimizer.step()
    #             optimizer.zero_grad()

    def loss_function(self, pred, target):
        return self.bce_loss(pred, target)

    def permutation_loss(self, pred, target):
        return self.mse_loss(pred, target)
    
    def valid_function(self, pred, target):
        _, pred_indices = pred.max(1)
        _, target_indices = target.max(1)
        return 1-torch.all(torch.eq(pred_indices, target_indices)).type(torch.float32)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        if optimizer_idx != 3:
            pred_sum, real_sum = self.forward(x, optimizer_idx)
            permutation_loss = self.permutation_loss(pred_sum, real_sum)
            logs = {'permutation_loss': permutation_loss}
            return {'loss': permutation_loss, 'log': logs}

        else:
            outputs = self.forward(x, optimizer_idx)
            loss = self.loss_function(outputs, y)
            logs = {'train_loss': loss}
            return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x, 3)
        loss = self.valid_function(outputs, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x, 3)
        loss = self.valid_function(outputs, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}