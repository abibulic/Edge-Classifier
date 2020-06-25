from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from graph_network import modules
from graph_network import utils_torch


class MLP_edge_model(torch.nn.Module):
    def __init__(self, network_config):
        super(MLP_edge_model, self).__init__()
        self.mlp = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(network_config["EDGE_IN"],network_config["EDGE_IN"]),
                    torch.nn.Linear(network_config["EDGE_IN"], network_config["EDGE_OUT"]),
                    torch.nn.Dropout(),
                    torch.nn.ReLU(),
                    torch.nn.Linear(network_config["EDGE_OUT"], network_config["EDGE_OUT"]),
                    torch.nn.Dropout(),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(network_config["EDGE_OUT"])
                    )
 
    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_node_model(torch.nn.Module):
    def __init__(self, network_config):
        super(MLP_node_model, self).__init__()
        self.mlp = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(network_config["NODE_IN"],network_config["NODE_IN"]),
                    torch.nn.Linear(network_config["NODE_IN"], network_config["NODE_OUT"]),
                    torch.nn.Dropout(),
                    torch.nn.ReLU(),
                    torch.nn.Linear(network_config["NODE_OUT"], network_config["NODE_OUT"]),
                    torch.nn.Dropout(),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(network_config["NODE_OUT"])
                    )
 
    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_global_model(torch.nn.Module):
    def __init__(self, network_config):
        super(MLP_global_model, self).__init__()
        self.mlp = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(network_config["GLOBAL_IN"],network_config["GLOBAL_IN"]),
                    torch.nn.Linear(network_config["GLOBAL_IN"], network_config["GLOBAL_OUT"]),
                    torch.nn.Dropout(),
                    torch.nn.ReLU(),
                    torch.nn.Linear(network_config["GLOBAL_OUT"], network_config["GLOBAL_OUT"]),
                    torch.nn.Dropout(),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(network_config["GLOBAL_OUT"])
                    )
 
    def forward(self, inputs):
        return self.mlp(inputs)

class GraphEncoder(torch.nn.Module):
    def __init__(self, network_config, block_config):
        super(GraphEncoder, self).__init__()
        self._network = modules.GraphNetwork(edge_model_fn=MLP_edge_model(network_config), 
                                            node_model_fn=MLP_node_model(network_config),
                                            global_model_fn=MLP_global_model(network_config),
                                            edge_block_opt=block_config["EDGE_BLOCK_OPT"],
                                            node_block_opt=block_config["NODE_BLOCK_OPT"],
                                            global_block_opt=block_config["GLOBAL_BLOCK_OPT"])
    def forward(self, inputs):
        return self._network(inputs)

class Latent(torch.nn.Module):
    def __init__(self, network_config):
        super(Latent, self).__init__()
        self.fc1 = torch.nn.Linear(network_config["encoder2"]["GLOBAL_OUT"], 400)
        self.fc2 = torch.nn.Linear(400, 100)
        self.fc31 = torch.nn.Linear(100, 20)
        self.fc32 = torch.nn.Linear(100, 20)
        self.fc4 = torch.nn.Linear(20, 100)
        self.fc5 = torch.nn.Linear(100, 400)
        self.fc6 = torch.nn.Linear(400, network_config["encoder2"]["GLOBAL_OUT"])

    def encode(self, x):
        h1 = torch.nn.functional.selu(self.fc1(x))
        h2 = torch.nn.functional.selu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.nn.functional.selu(self.fc4(z))
        h4 = torch.nn.functional.selu(self.fc5(h3))
        return self.fc6(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GraphDecoder_depricated(torch.nn.Module):
    def __init__(self, graph_options):
        super(GraphDecoder, self).__init__()
        self._edge_decode = torch.nn.Sequential(
                            torch.nn.Linear(10, 20),
                            torch.nn.ReLU(),
                            torch.nn.Linear(20, 20),
                            #torch.nn.LayerNorm(20),
                            torch.nn.Linear(20, 10))

        self._node_decode = torch.nn.Sequential(
                            torch.nn.Linear(20, 20),
                            torch.nn.ReLU(),
                            torch.nn.Linear(20, 20),
                            #torch.nn.LayerNorm(20),
                            torch.nn.Linear(20, 10))
    def forward(self, _z):
        temp = torch.split(_z, 100, dim=1)
        _z1 = temp[0]
        _z2 = temp[1]
        return self._edge_decode(_z1.reshape(128*10,10)), self._node_decode(_z2.reshape(128*5,20))

class GraphDecoder(torch.nn.Module):
    def __init__(self, network_config, block_config, binar):
        super(GraphDecoder, self).__init__()
        self._network = modules.GraphNetwork(edge_model_fn=MLP_edge_model(network_config), 
                                            node_model_fn=MLP_node_model(network_config),
                                            global_model_fn=MLP_global_model(network_config),
                                            edge_block_opt=block_config["EDGE_BLOCK_OPT"],
                                            node_block_opt=block_config["NODE_BLOCK_OPT"],
                                            global_block_opt=block_config["GLOBAL_BLOCK_OPT"])
        self.sig = torch.nn.Sigmoid()
        self.binar = binar
    def forward(self, inputs):
        if self.binar:
            e = []
            x = self._network(inputs)
            e.append(self.sig(x.edges[:,:1]))
            e.append(x.edges[:,1:])
            e = torch.cat(e, axis=-1)
            return x.replace(edges=e)
        else:
            return self._network(inputs)

class GraphVAE(torch.nn.Module):
    def __init__(self, network_config, block_config):
        super(GraphVAE, self).__init__()
        self._encoder1 = GraphEncoder(network_config["encoder1"], block_config["encoder1"])
        self._encoder2 = GraphEncoder(network_config["encoder2"], block_config["encoder2"])
        self._encoder3 = GraphEncoder(network_config["encoder3"], block_config["encoder3"])
        self._latent = Latent(network_config)
        self._decoder1 = GraphDecoder(network_config["decoder1"], block_config["decoder1"], binar=False)
        self._decoder2 = GraphDecoder(network_config["decoder2"], block_config["decoder2"], binar=True)
        # self._decoder2 = GraphEncoder(network_config, block_config)
        # self._decoder3 = GraphEncoder(network_config, block_config)

    def forward(self, graph, graph_placeholder):
        # z = self._encoder1(graph)
        # z = self._encoder2(z)
        # # z = self._encoder3(z)
        # _z, mu, logvar = self._latent(z.globals)

        _z = torch.ones(128,693).to(torch.device('cuda:0')) # za testiranje
        mu = torch.ones(128,693).to(torch.device('cuda:0'))
        logvar = torch.ones(128,693).to(torch.device('cuda:0'))

        x = self._decoder1(graph_placeholder.replace(globals=_z))
        x = self._decoder2(x)

        # reconstruct_edges, reconstruct_nodes = self._decoder1(_z)
        # x = self._decoder2(graph_placeholder.replace(edges=reconstruct_edges, nodes=reconstruct_nodes))
        # x = self._decoder3(x)
        return x, mu, logvar

########################################################################################################################################################################
## mali vae ##
##############

EDGE_BLOCK_OPT1 = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}
NODE_BLOCK_OPT1 = {
    "use_received_edges": False,
    "use_sent_edges": False,
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}
GLOBAL_BLOCK_OPT1 = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

EDGE_BLOCK_OPT2 = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": True,
}
NODE_BLOCK_OPT2 = {
    "use_received_edges": False,
    "use_sent_edges": False,
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}
GLOBAL_BLOCK_OPT2 = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}

class Encode(torch.nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self._network = modules.GraphNetwork(edge_model_fn=torch.nn.Sequential(
                                                        #torch.nn.BatchNorm1d(31, 31),
                                                        torch.nn.Linear(31, 25),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(25, 20),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(20, 10),
                                                        ), 
                                            node_model_fn=torch.nn.Sequential(
                                                        #torch.nn.BatchNorm1d(18, 18),
                                                        torch.nn.Linear(23, 18),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(18, 10),
                                                        ),
                                            global_model_fn=torch.nn.Sequential(
                                                        torch.nn.Linear(20, 15),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(15, 15),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(15, 10),
                                                        ),
                                            edge_block_opt=EDGE_BLOCK_OPT1,
                                            node_block_opt=NODE_BLOCK_OPT1,
                                            global_block_opt=GLOBAL_BLOCK_OPT1)
    def forward(self, inputs):
        return self._network(inputs)

class Lat(torch.nn.Module):
    def __init__(self):
        super(Lat, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(10, 20)

        self.fc3 = torch.nn.Linear(10, 20)
        self.fc4 = torch.nn.Linear(10, 20)
        
    def encode(self, x):
        return self.fc1(x.nodes), self.fc2(x.nodes), self.fc3(x.edges), self.fc4(x.edges)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu_n, logvar_n, mu_e, logvar_e = self.encode(x)
        z_n = self.reparameterize(mu_n, logvar_n)
        z_e = self.reparameterize(mu_e, logvar_e)
        return z_n, mu_n, logvar_n, z_e, mu_e, logvar_e

class Decode(torch.nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self._edge_decode = torch.nn.Sequential(
                            torch.nn.Linear(20, 10),
                            torch.nn.SELU(),
                            torch.nn.Linear(10, 5))

        self._node_decode = torch.nn.Sequential(
                            torch.nn.Linear(20, 16),
                            torch.nn.SELU(),
                            torch.nn.Linear(16, 13))

    def forward(self, z_n, z_e):
        return self._edge_decode(z_e), self._node_decode(z_n)


class gvae(torch.nn.Module):
    def __init__(self):
        super(gvae, self).__init__()
        self._encoder = Encode()
        self._latent = Lat()
        self._decoder = Decode()

    def forward(self, inputs):
        x = self._encoder(inputs)
        z_n, mu_n, logvar_n, z_e, mu_e, logvar_e = self._latent(x)
        e, n = self._decoder(z_n, z_e)
        return x.replace(edges=e, nodes=n), mu_n, logvar_n, mu_e, logvar_e


########################################################################################################################################################################
## regresija ##
###############

EDGE_BLOCK_OPT1 = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}
NODE_BLOCK_OPT1 = {
    "use_received_edges": False,
    "use_sent_edges": False,
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}
GLOBAL_BLOCK_OPT1 = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

EDGE_BLOCK_OPT2 = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": True,
}
NODE_BLOCK_OPT2 = {
    "use_received_edges": False,
    "use_sent_edges": False,
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}
GLOBAL_BLOCK_OPT2 = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}
class GraphRegression(torch.nn.Module):
    def __init__(self):
        super(GraphRegression, self).__init__()
        self.graph1 = modules.GraphNetwork(edge_model_fn=torch.nn.Sequential(
                                                        torch.nn.BatchNorm1d(31, 31),
                                                        torch.nn.Linear(31, 20),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(20, 10),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(10, 5),
                                                        torch.nn.SELU(),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ), 
                                            node_model_fn=torch.nn.Sequential(
                                                        torch.nn.BatchNorm1d(18, 18),
                                                        torch.nn.Linear(18, 15),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(15, 13),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ),
                                            global_model_fn=torch.nn.Sequential(
                                                        torch.nn.Linear(18, 18),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(18, 15),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(15, 12),
                                                        #torch.nn.ReLU(),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ),
                                            edge_block_opt=EDGE_BLOCK_OPT1,
                                            node_block_opt=NODE_BLOCK_OPT1,
                                            global_block_opt=GLOBAL_BLOCK_OPT1)

        self.graph11 = modules.GraphNetwork(edge_model_fn=torch.nn.Sequential(
                                                        #torch.nn.BatchNorm1d(36, 36),
                                                        torch.nn.Linear(36, 128),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(128, 256),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(256, 128),
                                                        torch.nn.SELU(),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ), 
                                            node_model_fn=torch.nn.Sequential(
                                                        #torch.nn.BatchNorm1d(19, 19),
                                                        torch.nn.Linear(145, 100),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(100, 50),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ),
                                            global_model_fn=torch.nn.Sequential(
                                                        torch.nn.Linear(178, 100),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(100, 50),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(50, 7),
                                                        #torch.nn.ReLU(),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ),
                                            edge_block_opt=EDGE_BLOCK_OPT1,
                                            node_block_opt=NODE_BLOCK_OPT1,
                                            global_block_opt=GLOBAL_BLOCK_OPT1)
        
        self.graph2 = modules.GraphNetwork(edge_model_fn=torch.nn.Sequential(
                                                        torch.nn.BatchNorm1d(145, 145),
                                                        torch.nn.Linear(145, 120),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(120, 110),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(110, 100),
                                                        torch.nn.SELU(),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ), 
                                            node_model_fn=torch.nn.Sequential(
                                                        torch.nn.BatchNorm1d(185, 185),
                                                        torch.nn.Linear(185, 160),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(160, 120),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(120, 100),
                                                        torch.nn.SELU(),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ),
                                            global_model_fn=torch.nn.Sequential(
                                                        torch.nn.Linear(250, 200),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(200, 100),
                                                        #torch.nn.Dropout(),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(100, 50),
                                                        torch.nn.SELU(),
                                                        torch.nn.Linear(50, 7),
                                                        #torch.nn.ReLU(),
                                                        #torch.nn.LayerNorm(out_size)
                                                        ),
                                            edge_block_opt=EDGE_BLOCK_OPT2,
                                            node_block_opt=NODE_BLOCK_OPT2,
                                            global_block_opt=GLOBAL_BLOCK_OPT2)
    def forward(self, inputs):
        x = self._graph11(inputs)
        #x = self._graph2(x)
        return x.globals


########################################################################################################################################################################
## novi koncept za mpnn ##
##########################
from gn.blocks import *


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

import time

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()

        self.permutate_nodes = torch.nn.Sequential(
                                torch.nn.Linear(26, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 26),
                                )
        
        self.update_edges = torch.nn.Sequential(
                                torch.nn.Linear(31, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 31),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(31, 5),
                                )

        self.edges_aggregator = CollectNeighboursAndEdgesToNodes()
        self.neighbours_collector = NeighboursToNodesCollector()
        self.edges_collector = EdgesToNodesCollector()

        self.permutate_neighbours = torch.nn.Sequential(
                                torch.nn.Linear(52, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 52),
                                )

        self.permutate_edges = torch.nn.Sequential(
                                torch.nn.Linear(20, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 20),
                                )
        
        self.update_nodes = torch.nn.Sequential(
                                torch.nn.Linear(85, 50),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(50, 25),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(25, 13),
                                )

        self.predict= torch.nn.Sequential(
                                torch.nn.Linear(85, 50),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(50, 25),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(25, 12),
                                )

        self.aggregate_all = NodesToGlobalsAggregator()

    def forward(self, graph):
        
        # 2 noda za svaki edge
        collect_nodes_for_edges = []
        collect_nodes_for_edges.append(broadcast_receiver_nodes_to_edges(graph))
        collect_nodes_for_edges.append(broadcast_sender_nodes_to_edges(graph))
        collected_nodes = torch.cat(collect_nodes_for_edges, axis=-1)

        h1_nodes = self.permutate_nodes(collected_nodes)

        #h1_edges =[]
        h1_edges = graph.edges
        for t in range(3):
            temp = torch.cat([h1_edges, h1_nodes], axis=-1)
            h1_edges = self.update_edges(temp)

        neighbours_of_nodes, edges_of_nodes = self.edges_aggregator(graph.replace(edges=h1_edges))
        # neighbours_of_nodes = self.neighbours_collector(graph)
        # edges_of_nodes = self.edges_collector(graph.replace(edges=h1_edges))
        
        h_neighbours = self.permutate_neighbours(neighbours_of_nodes)
        h2_edges = self.permutate_edges(edges_of_nodes)

        h2_nodes = graph.nodes
        for t in range(3):
            temp2 = torch.cat([h2_nodes, h_neighbours, h2_edges], axis=-1)
            h2_nodes = self.update_nodes(temp2)

        temp3 = self.aggregate_all(graph.replace(nodes=torch.cat([h2_nodes, h_neighbours, h2_edges], axis=-1)))
        out = self.predict(temp3.to(torch.device('cuda')))

        return out


########################################################################################################################################################################
## MNIST Graph ##
##########################

class MNIST_Graph(torch.nn.Module):
    def __init__(self):
        super(MNIST_Graph, self).__init__()

        self.permutate_nodes = torch.nn.Sequential(
                                torch.nn.Linear(2, 10),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(10, 10),
                                )
        
        self.update_edges = torch.nn.Sequential(
                                torch.nn.Linear(11, 5),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(5, 1),
                                )

        self.edges_aggregator = EdgesToNodesAggregator()

        self.permutate_edges = torch.nn.Sequential(
                                torch.nn.Linear(1, 5),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(5, 10),
                                )
        
        self.update_nodes = torch.nn.Sequential(
                                torch.nn.Linear(11, 5),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(5, 1),
                                )

        self.predict= torch.nn.Sequential(
                                torch.nn.Linear(11, 10),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(10, 10),
                                )

        self.aggregate_all = NodesToGlobalsAggregator()

    def forward(self, graph):
        
        # 2 noda za svaki edge
        collect_nodes_for_edges = []
        collect_nodes_for_edges.append(broadcast_receiver_nodes_to_edges(graph))
        collect_nodes_for_edges.append(broadcast_sender_nodes_to_edges(graph))
        collected_nodes = torch.cat(collect_nodes_for_edges, axis=-1)

        h1_nodes = self.permutate_nodes(collected_nodes)

        #h1_edges =[]
        h1_edges = graph.edges
        for t in range(3):
            temp = torch.cat([h1_edges, h1_nodes], axis=-1)
            h1_edges = self.update_edges(temp)
        
        edges_of_nodes = self.edges_aggregator(graph.replace(edges=h1_edges)).to(torch.device('cuda'))
        
        h2_edges = self.permutate_edges(edges_of_nodes)

        h2_nodes = graph.nodes
        for t in range(3):
            temp2 = torch.cat([h2_nodes, h2_edges], axis=-1)
            h2_nodes = self.update_nodes(temp2)

        temp3 = self.aggregate_all(graph.replace(nodes=torch.cat([h2_nodes, h2_edges], axis=-1)))
        out = self.predict(temp3.to(torch.device('cuda')))

        return torch.nn.functional.softmax(out)


########################################################################################################################################################################
## novi koncept za VAE ##
##########################

class Lat2(torch.nn.Module):
    def __init__(self):
        super(Lat, self).__init__()
        self.fc1 = torch.nn.Linear(25, 20)
        self.fc2 = torch.nn.Linear(25, 20)

        self.fc3 = torch.nn.Linear(25, 20)
        self.fc4 = torch.nn.Linear(25, 20)
        
    def encode(self, x):
        return self.fc1(x.nodes), self.fc2(x.nodes), self.fc3(x.edges), self.fc4(x.edges)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu_n, logvar_n, mu_e, logvar_e = self.encode(x)
        z_n = self.reparameterize(mu_n, logvar_n)
        z_e = self.reparameterize(mu_e, logvar_e)
        return z_n, mu_n, logvar_n, z_e, mu_e, logvar_e

class Test2(torch.nn.Module):
    def __init__(self):
        super(Test2, self).__init__()

        self.permutate_nodes = torch.nn.Sequential(
                                torch.nn.Linear(26, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 26),
                                )
        
        self.update_edges = torch.nn.Sequential(
                                torch.nn.Linear(31, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 31),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(31, 5),
                                )

        self.neighbours_and_edges_aggregator = CollectNeighboursAndEdgesToNodes()

        self.permutate_neighbours = torch.nn.Sequential(
                                torch.nn.Linear(52, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 52),
                                )

        self.permutate_edges = torch.nn.Sequential(
                                torch.nn.Linear(20, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 64),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(64, 20),
                                )
        
        self.update_nodes = torch.nn.Sequential(
                                torch.nn.Linear(85, 50),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(50, 25),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(25, 13),
                                )

        self.aggregate_all = NodesToGlobalsAggregator()

        self.encode= torch.nn.Sequential(
                                torch.nn.Linear(85, 85),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(85, 60),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(60, 50),
                                )

    def forward(self, graph):
        
        # 2 noda za svaki edge
        collect_nodes_for_edges = []
        collect_nodes_for_edges.append(broadcast_receiver_nodes_to_edges(graph))
        collect_nodes_for_edges.append(broadcast_sender_nodes_to_edges(graph))
        collected_nodes = torch.cat(collect_nodes_for_edges, axis=-1)

        h1_nodes = self.permutate_nodes(collected_nodes)

        #h1_edges =[]
        h1_edges = graph.edges
        for t in range(3):
            temp = torch.cat([h1_edges, h1_nodes], axis=-1)
            h1_edges = self.update_edges(temp)
        
        neighbours_of_nodes, edges_of_nodes = self.neighbours_and_edges_aggregator(graph.replace(edges=h1_edges))
        
        h_neighbours = self.permutate_neighbours(neighbours_of_nodes)
        h2_edges = self.permutate_edges(edges_of_nodes)

        h2_nodes = graph.nodes
        for t in range(3):
            temp2 = torch.cat([h2_nodes, h_neighbours, h2_edges], axis=-1)
            h2_nodes = self.update_nodes(temp2)

        temp3 = self.aggregate_all(graph.replace(nodes=torch.cat([h2_nodes, h_neighbours, h2_edges], axis=-1)))
        z = self.encode(temp3.to(torch.device('cuda')))

        temp = torch.split(_z, 25, dim=1)


        return out


########################################################################################################################################################################
## ARC Graph ##
##########################

class ARC(torch.nn.Module):
    def __init__(self):
        super(ARC, self).__init__()

        self.permutate_nodes = torch.nn.Sequential(
                                torch.nn.Linear(6, 16),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 32),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                )
        
        self.update_edges = torch.nn.Sequential(
                                torch.nn.Linear(19, 32),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 3),
                                )

        self.edges_collector = EdgesToNodesAggregator()

        self.permutate_edges = torch.nn.Sequential(
                                torch.nn.Linear(3, 16),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 32),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                )
        
        self.update_nodes = torch.nn.Sequential(
                                torch.nn.Linear(19, 32),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 3),
                                )

        self.aggregate_all = NodesToGlobalsAggregator()

        self.predict= torch.nn.Sequential(
                                torch.nn.Linear(19, 10),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(10, 10),
                                #torch.nn.Dropout(),
                                torch.nn.SELU(),
                                torch.nn.Linear(10, 9),
                                )

    def forward(self, graph):
        
        # 2 noda za svaki edge
        collect_nodes_for_edges = []
        collect_nodes_for_edges.append(broadcast_receiver_nodes_to_edges(graph))
        collect_nodes_for_edges.append(broadcast_sender_nodes_to_edges(graph))
        collected_nodes = torch.cat(collect_nodes_for_edges, axis=-1)

        h1_nodes = self.permutate_nodes(collected_nodes)

        #h1_edges =[]
        h1_edges = graph.edges
        for t in range(3):
            temp = torch.cat([h1_edges, h1_nodes], axis=-1)
            h1_edges = self.update_edges(temp)

        edges_of_nodes = self.edges_collector(graph.replace(edges=h1_edges))
        
        h2_edges = self.permutate_edges(edges_of_nodes.to(torch.device('cuda')))

        h2_nodes = graph.nodes
        for t in range(3):
            temp2 = torch.cat([h2_nodes, h2_edges], axis=-1)
            h2_nodes = self.update_nodes(temp2)

        temp3 = self.aggregate_all(graph.replace(nodes=torch.cat([h2_nodes, h2_edges], axis=-1)))
        out = self.predict(temp3.to(torch.device('cuda')))

        return out


########################################################################################################################################################################
## ARC2 Graph ##
##########################

class ARC2(torch.nn.Module):
    def __init__(self):
        super(ARC2, self).__init__()

        self.permutate_nodes = torch.nn.Sequential(
                                torch.nn.Linear(6, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                )
        
        self.update_edges = torch.nn.Sequential(
                                torch.nn.Linear(19, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 8),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(8, 3),
                                )

        self.edges_collector = EdgesToNodesAggregator()

        self.permutate_edges = torch.nn.Sequential(
                                torch.nn.Linear(3, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                )
        
        self.update_nodes = torch.nn.Sequential(
                                torch.nn.Linear(19, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 8),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(8, 3),
                                )

        self.aggregate_all = NodesToGlobalsAggregator()

        self.predict= torch.nn.Sequential(
                                torch.nn.Linear(19, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 12),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(12, 8),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(8, 9),
                                )

    def forward(self, graph):
        
        # 2 noda za svaki edge
        collect_nodes_for_edges = []
        collect_nodes_for_edges.append(broadcast_receiver_nodes_to_edges(graph))
        collect_nodes_for_edges.append(broadcast_sender_nodes_to_edges(graph))
        collected_nodes = torch.cat(collect_nodes_for_edges, axis=-1)

        h1_nodes = self.permutate_nodes(collected_nodes)

        #h1_edges =[]
        h1_edges = graph.edges
        for t in range(10):
            temp = torch.cat([h1_edges, h1_nodes], axis=-1)
            h1_edges = self.update_edges(temp)

        edges_of_nodes = self.edges_collector(graph.replace(edges=h1_edges))
        
        h2_edges = self.permutate_edges(edges_of_nodes.to(torch.device('cuda')))

        h2_nodes = graph.nodes
        for t in range(10):
            temp2 = torch.cat([h2_nodes, h2_edges], axis=-1)
            h2_nodes = self.update_nodes(temp2)

        temp3 = self.aggregate_all(graph.replace(nodes=torch.cat([h2_nodes, h2_edges], axis=-1)))
        out = self.predict(temp3.to(torch.device('cuda')))

        return out


########################################################################################################################################################################
## ARC3 Graph ##
##########################

EDGE_BLOCK_OPT_ARC = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}
NODE_BLOCK_OPT_ARC = {
    "use_received_edges": False,
    "use_sent_edges": False,
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}
GLOBAL_BLOCK_OPT_ARC = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

class ARC3(torch.nn.Module):
    def __init__(self):
        super(ARC3, self).__init__()

        self.permutate_nodes = torch.nn.Sequential(
                                torch.nn.Linear(3, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 3)
                                )
        
        self.permutate_edges = torch.nn.Sequential(
                                torch.nn.Linear(3, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 3)
                                )

        self.core = modules.GraphNetwork(edge_model_fn=torch.nn.Sequential(
                                        torch.nn.Linear(18, 18),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(18, 12),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(12, 6)
                                        ), 
                                    node_model_fn=torch.nn.Sequential(
                                        torch.nn.Linear(12, 12),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(12, 12),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(12, 6)
                                        ),
                                    global_model_fn=torch.nn.Sequential(
                                        torch.nn.Linear(12, 12),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(12, 12),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(12, 9)
                                        ),
                                    edge_block_opt=EDGE_BLOCK_OPT_ARC,
                                    node_block_opt=NODE_BLOCK_OPT_ARC,
                                    global_block_opt=GLOBAL_BLOCK_OPT_ARC)

        self.predict_nodes = torch.nn.Sequential(
                                torch.nn.Linear(3, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 32),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(32, 16),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(16, 10)
                                )

    def forward(self, graph):
        
        latent_nodes = self.permutate_nodes(graph.nodes)
        latent_edges = self.permutate_edges(graph.edges)

        latent0_nodes = latent_nodes
        latent0_edges = latent_edges

        output_ops = []

        for _ in range(3):
            nodes_input = torch.cat([latent0_nodes, latent_nodes], axis=-1) #6
            edges_input = torch.cat([latent0_edges, latent_edges], axis=-1) #6
            latent = self.core(graph.replace(nodes=nodes_input, edges=edges_input))
            latent_nodes = graph.nodes
            latent_edges = graph.edges
            decoded_nodes = self.predict_nodes(latent_nodes)

        return decoded_nodes, latent.globals

########################################################################################################################################################################
## ARC4 Graph ##
##########################

EDGE_BLOCK_OPT_ARC4 = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}
NODE_BLOCK_OPT_ARC4 = {
    "use_received_edges": False,
    "use_sent_edges": False,
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}
GLOBAL_BLOCK_OPT_ARC4 = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

class ARC4(torch.nn.Module):
    def __init__(self):
        super(ARC4, self).__init__()

        self.permutate_nodes = torch.nn.Sequential(
                                torch.nn.Linear(3, 10),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(10, 10)
                                )
        
        self.permutate_edges = torch.nn.Sequential(
                                torch.nn.Linear(3, 10),
                                #torch.nn.Dropout(0.1),
                                torch.nn.SELU(),
                                torch.nn.Linear(10, 10)
                                )

        self.core = modules.GraphNetwork(edge_model_fn=torch.nn.Sequential(
                                        torch.nn.Linear(30, 64),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(64, 32),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(32, 10)
                                        ), 
                                    node_model_fn=torch.nn.Sequential(
                                        torch.nn.Linear(20, 64),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(64, 32),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(32, 10)
                                        ),
                                    global_model_fn=torch.nn.Sequential(
                                        torch.nn.Linear(20, 64),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(64, 32),
                                        #torch.nn.Dropout(0.1),
                                        torch.nn.SELU(),
                                        torch.nn.Linear(32, 10)
                                        ),
                                    edge_block_opt=EDGE_BLOCK_OPT_ARC4,
                                    node_block_opt=NODE_BLOCK_OPT_ARC4,
                                    global_block_opt=GLOBAL_BLOCK_OPT_ARC4)

    def forward(self, graph):
        
        latent_nodes = self.permutate_nodes(graph.nodes)
        latent_edges = self.permutate_edges(graph.edges)

        output_ops = []

        latent = graph.replace(nodes=latent_nodes,edges=latent_edges)

        for _ in range(3):
            latent = self.core(latent)

        return latent.nodes, latent.globals