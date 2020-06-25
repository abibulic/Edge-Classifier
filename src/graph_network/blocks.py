from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from graph_network import graphs
from graph_network import utils_torch

NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

def _validate_graph(graph, mandatory_fields, additional_message=None):
    for field in mandatory_fields:
        if getattr(graph, field) is None:
            message = "`{}` field cannot be None".format(field)
            if additional_message:
                message += " " + format(additional_message)
            message += "."
            raise ValueError(message)

def _validate_broadcasted_graph(graph, from_field, to_field):
    additional_message = "when broadcasting {} to {}".format(from_field, to_field)
    _validate_graph(graph, [from_field, to_field], additional_message)

def broadcast_globals_to_edges(graph, name="broadcast_globals_to_edges"):
    _validate_broadcasted_graph(graph, GLOBALS, N_EDGE)
    return torch.repeat_interleave(graph.globals, graph.n_edge[0].item(), 0)

def broadcast_globals_to_nodes(graph, name="broadcast_globals_to_nodes"):
    _validate_broadcasted_graph(graph, GLOBALS, N_NODE)
    return torch.repeat_interleave(graph.globals, graph.n_node[0].item(), 0)

def broadcast_sender_nodes_to_edges(graph, name="broadcast_sender_nodes_to_edges"):

    """Kreira tensor velicine n_edges * node_shape preslikavanjem vrijednosti 
    iz graph.nodes po redosljedu indeksiranom u graph.senders"""

    _validate_broadcasted_graph(graph, NODES, SENDERS)
    return torch.index_select(graph.nodes, 0, graph.senders)

def broadcast_receiver_nodes_to_edges(graph, name="broadcast_receiver_nodes_to_edges"):

    """Kreira tensor velicine n_edges * node_shape preslikavanjem vrijednosti 
    iz graph.nodes po redosljedu indeksiranom u graph.recievers"""

    _validate_broadcasted_graph(graph, NODES, RECEIVERS)
    return torch.index_select(graph.nodes, 0, graph.receivers)


class receivedEdgesToNodesAggregator(torch.nn.Module):
    def __init__(self):
        """Za svaki node zbroji sve njegove ulazne edgeve."""
        super(receivedEdgesToNodesAggregator, self).__init__()
    def forward(self, graph):
        #num_nodes = torch.sum(graph.n_node)
        num_nodes = graph.nodes.shape[0]
        num_edge_features = graph.edges.shape[-1]
        indices = graph.receivers
        return torch.zeros(num_nodes, num_edge_features).index_add(0, indices.to(torch.device('cpu')), graph.edges.to(torch.device('cpu')))


class sentEdgesToNodesAggregator(torch.nn.Module):
    def __init__(self):
        """Za svaki node zbroji sve njegove izlazne edgeve"""
        super(sentEdgesToNodesAggregator, self).__init__()
    def forward(self, graph):
        #num_nodes = torch.sum(graph.n_node)
        num_nodes = graph.nodes.shape[0]
        num_edge_features = graph.edges.shape[-1]
        indices = graph.senders
        return torch.zeros(num_nodes, num_edge_features).index_add(0, indices.to(torch.device('cpu')), graph.edges.to(torch.device('cpu')))


class EdgesToNodesAggregator(torch.nn.Module):
    def __init__(self):
        """Za svaki node zbroji sve njegove edgeve"""
        super(EdgesToNodesAggregator, self).__init__()
    def forward(self, graph):
        _validate_graph(graph, (EDGES, SENDERS, RECEIVERS,),
                        additional_message="when aggregating from edges.")
        #num_nodes = torch.sum(graph.n_node)
        num_nodes = graph.nodes.shape[0]
        num_edge_features = graph.edges.shape[-1]
        return (torch.zeros(num_nodes, num_edge_features).index_add(0, graph.senders.to(torch.device('cpu')), graph.edges.to(torch.device('cpu'))) +
                torch.zeros(num_nodes, num_edge_features).index_add(0, graph.receivers.to(torch.device('cpu')), graph.edges.to(torch.device('cpu'))))


class EdgesToGlobalsAggregator(torch.nn.Module):
    def __init__(self):
        super(EdgesToGlobalsAggregator, self).__init__()
    def forward(self, graph):
        _validate_graph(graph, (EDGES,),
                        additional_message="when aggregating from edges.")
        num_graphs = graph.n_node.shape[0]
        num_edge_features = graph.edges.shape[-1]
        num_edges = graph.n_edge.type(torch.LongTensor)
        graph_index = torch.arange(0, num_graphs)
        indices = torch.repeat_interleave(graph_index, num_edges)
        return torch.zeros(num_graphs, num_edge_features).index_add(0, indices.to(torch.device('cpu')), graph.edges.to(torch.device('cpu')))


class NeighboursToNodesCollector(torch.nn.Module):
    def __init__(self):
        super(NeighboursToNodesCollector, self).__init__()
    def forward(self, graph):
        """nađe sve unikatne edgeve (unikatni brojevi iz sendera i receivera zajedno)"""
        unique_edges_count = torch.unique(torch.cat((graph.senders, graph.receivers)), sorted=True, return_counts=True)
        """vrati inekse sa brojem unikatnih edgeva"""
        num_nodes = graph.nodes.shape[0]
        num_node_features = graph.nodes.shape[-1]
        num_edge_features = graph.edges.shape[-1]

        graph_index = torch.arange(0, num_nodes)
        indices = torch.repeat_interleave(graph_index.to(torch.device('cuda')), unique_edges_count[1])
        """nalazi nodove koji su vezani za indices (za svaki node u index postoji node index2)"""
        indices2 = []
        for x in range(num_nodes):
            indices2 += graph.receivers[(graph.senders==x).nonzero().view(-1).tolist()].tolist() + graph.senders[(graph.receivers==x).nonzero().view(-1).tolist()].tolist()
        indices2 = torch.tensor(indices2).view(-1, 1).repeat(1, num_node_features).view(-1, num_node_features)
        """skupi sve po indices2 i splitaj po broju edgeva"""
        neighbours = torch.gather(graph.nodes, 0 ,indices2.to(torch.device('cuda')))
        neighbours = torch.split(neighbours, unique_edges_count[1].tolist(), dim=0)

        """proširi sa nulama i reshape"""
        return torch.cat([torch.cat((x, (torch.zeros(4-x.shape[0], num_node_features).to(torch.device('cuda'))))) for x in neighbours]).view(num_nodes, -1)


class EdgesToNodesCollector(torch.nn.Module):
    def __init__(self):
        super(EdgesToNodesCollector, self).__init__()
    def forward(self, graph):
        num_nodes = graph.nodes.shape[0]
        num_edges = graph.edges.shape[0]
        num_edge_features = graph.edges.shape[-1]
        graph_index = torch.arange(0, num_edges)
        """kreiraj matricu povezanosti edgeva s nodovima"""
        adj = torch.zeros((num_edges, num_edges, num_edge_features)).to(torch.device('cuda'))
        adj[graph.senders, graph_index] = 1.
        adj[graph.receivers, graph_index] = 1.
        """ispuni matricu vrijednostima edgeva"""
        adj = graph.edges * adj
        adj = adj[:(num_nodes-num_edges), :, :]
        """TODO: Ovo je najnepreglednije ikad (Ubiti samo posloži tenzor u [num_nodes x (4*num_edge_features)] i doda nule redke ako je manje od 4 edga po nodu)"""
        return torch.cat([torch.cat((x[abs(x.sum(dim=1))!=0].view(-1,num_edge_features), (torch.zeros(4-x[abs(x.sum(dim=1))!=0].view(-1,num_edge_features).shape[0], num_edge_features).to(torch.device('cuda'))))) for x in adj]).view(num_nodes, -1)


class NodesToGlobalsAggregator(torch.nn.Module):
    def __init__(self):
        super(NodesToGlobalsAggregator, self).__init__()
    def forward(self, graph):
        _validate_graph(graph, (NODES,),
                        additional_message="when aggregating from nodes.")
        num_graphs = graph.n_node.shape[0]
        num_node_features = graph.nodes.shape[-1]
        num_nodes = graph.n_node.type(torch.LongTensor)
        graph_index = torch.arange(0, num_graphs)
        indices = torch.repeat_interleave(graph_index, num_nodes)
        return torch.zeros(num_graphs, num_node_features).index_add(0, indices.to(torch.device('cpu')), graph.nodes.to(torch.device('cpu')))
    

class EdgeBlock(torch.nn.Module):

    def __init__(self,
                edge_model_fn,
                use_edges=True,
                use_receiver_nodes=True,
                use_sender_nodes=True,
                use_globals=False):

        super(EdgeBlock, self).__init__()

        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
            raise ValueError("At least one of use_edges, use_sender_nodes, "
                            "use_receiver_nodes or use_globals must be True.")
        
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals

        self._edge_model = edge_model_fn

    def forward(self, graph):
        _validate_graph(graph, (SENDERS, RECEIVERS, N_EDGE), " when using an EdgeBlock")

        edges_to_collect = []

        if self._use_edges:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            edges_to_collect.append(graph.edges)

        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))

        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))

        if self._use_globals:
            edges_to_collect.append(broadcast_globals_to_edges(graph))

        collected_edges = torch.cat(edges_to_collect, axis=-1)
        updated_edges = self._edge_model(collected_edges)
        #print(updated_edges[0])
        #print(updated_edges[1])
        return graph.replace(edges=updated_edges)


class NodeBlock(torch.nn.Module):

    def __init__(self,
                node_model_fn,
                use_received_edges=False,
                use_sent_edges=False,
                use_edges = True,
                use_nodes=True,
                use_globals=False,
                ):

        super(NodeBlock, self).__init__()

        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
            raise ValueError("At least one of use_received_edges, use_sent_edges, "
                            "use_nodes or use_globals must be True.")
        
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        self._node_model = node_model_fn

        if self._use_received_edges:
            self._received_edges_aggregator = receivedEdgesToNodesAggregator()
        if self._use_sent_edges:    
            self._sent_edges_aggregator = sentEdgesToNodesAggregator()
        if self._use_edges:    
            self._edge_aggregator = EdgesToNodesAggregator()

    def forward(self, graph):
        
        nodes_to_collect = []

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph).to(torch.device('cuda:0')))

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph).to(torch.device('cuda:0')))

        if self._use_edges:
            nodes_to_collect.append(self._edge_aggregator(graph).to(torch.device('cuda:0')))

        if self._use_nodes:
            _validate_graph(graph, (NODES,), "when use_nodes == True")
            nodes_to_collect.append(graph.nodes)

        if self._use_globals:
            nodes_to_collect.append(broadcast_globals_to_nodes(graph).to(torch.device('cuda:0')))

        collected_nodes = torch.cat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)
        #print(updated_nodes[0])
        #print(updated_nodes[1])
        return graph.replace(nodes=updated_nodes)

class GlobalBlock(torch.nn.Module):

    def __init__(self,
                global_model_fn,
                use_edges=True,
                use_nodes=True,
                use_globals=True,
                ):

        super(GlobalBlock, self).__init__()

        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, "
                             "use_nodes or use_globals must be True.")

        
        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        self._global_model = global_model_fn

        if self._use_edges:
            self._edges_aggregator = EdgesToGlobalsAggregator()
        if self._use_nodes:
            self._nodes_aggregator = NodesToGlobalsAggregator()

    def forward(self, graph):
        
        globals_to_collect = []

        if self._use_edges:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            globals_to_collect.append(self._edges_aggregator(graph).to(torch.device('cuda:0')))

        if self._use_nodes:
            _validate_graph(graph, (NODES,), "when use_nodes == True")
            globals_to_collect.append(self._nodes_aggregator(graph).to(torch.device('cuda:0')))

        if self._use_globals:
            _validate_graph(graph, (GLOBALS,), "when use_globals == True")
            globals_to_collect.append(graph.globals)

        collected_globals = torch.cat(globals_to_collect, axis=-1)
        updated_globals = self._global_model(collected_globals)
        #print(updated_globals)
        return graph.replace(globals=updated_globals)




        