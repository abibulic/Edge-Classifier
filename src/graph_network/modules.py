from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from graph_network import blocks

_DEFAULT_EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}

_DEFAULT_NODE_BLOCK_OPT = {
    "use_received_edges": False,
    "use_sent_edges": False,
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

_DEFAULT_GLOBAL_BLOCK_OPT = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

def _make_default_edge_block_opt(edge_block_opt):
    """Default options to be used in the EdgeBlock of a generic GraphNetwork."""
    edge_block_opt = dict(edge_block_opt.items()) if edge_block_opt else {}
    for k, v in _DEFAULT_EDGE_BLOCK_OPT.items():
        edge_block_opt[k] = edge_block_opt.get(k, v)
    return edge_block_opt


def _make_default_node_block_opt(node_block_opt):
    """Default options to be used in the NodeBlock of a generic GraphNetwork."""
    node_block_opt = dict(node_block_opt.items()) if node_block_opt else {}
    for k, v in _DEFAULT_NODE_BLOCK_OPT.items():
        node_block_opt[k] = node_block_opt.get(k, v)
    return node_block_opt


def _make_default_global_block_opt(global_block_opt):
    """Default options to be used in the GlobalBlock of a generic GraphNetwork."""
    global_block_opt = dict(global_block_opt.items()) if global_block_opt else {}
    for k, v in _DEFAULT_GLOBAL_BLOCK_OPT.items():
        global_block_opt[k] = global_block_opt.get(k, v)
    return global_block_opt

class GraphNetwork(torch.nn.Module):
    def __init__(self,
                edge_model_fn,
                node_model_fn,
                global_model_fn,
                edge_block_opt=None,
                node_block_opt=None,
                global_block_opt=None,
                ):
        super(GraphNetwork, self).__init__()
        edge_block_opt = _make_default_edge_block_opt(edge_block_opt)
        node_block_opt = _make_default_node_block_opt(node_block_opt)
        global_block_opt = _make_default_global_block_opt(global_block_opt)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=edge_model_fn, **edge_block_opt)
        self._node_block = blocks.NodeBlock(
            node_model_fn=node_model_fn, **node_block_opt)
        self._global_block = blocks.GlobalBlock(
            global_model_fn=global_model_fn, **global_block_opt)

    def forward(self, graph):
        return self._global_block(self._node_block(self._edge_block(graph)))

