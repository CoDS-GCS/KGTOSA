from typing import Optional, Union

import torch

from .GAT import GAT
from .GCN import GCN
from .SAGE import SAGEModel
from .RGCN import RGCN


def get_model(graphmodel: str,
              num_node_features: int,
              num_classes: int,
              hidden_channels: int,
              num_layers: int,
              heads: Optional[int],
              device: Union[torch.device, str],data=None,key2int=None):
    if graphmodel == 'rgcn':
        device = 'cpu'
        num_nodes_dict = {}
        for key, N in data.num_nodes_dict.items():
            num_nodes_dict[key2int[key]] = N
        #x_dict = {}
        #for key, x in data.x_dict.items():
        #    x_dict[key2int[key]] = x
        model = RGCN(128, hidden_channels, num_classes, num_layers,
                     0.5, num_nodes_dict, list(data.y_dict.keys()),
                     len(data.edge_index_dict.keys()))
    elif graphmodel == 'gcn':
        model = GCN(num_node_features=num_node_features,
                    num_classes=num_classes,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers)

    elif graphmodel == 'gat':
        model = GAT(in_channels=num_node_features,
                    hidden_channels=hidden_channels,
                    out_channels=num_classes,
                    num_layers=num_layers,
                    heads=heads)
    elif graphmodel == 'sage':
        model = SAGEModel(num_node_features=num_node_features,
                          num_classes=num_classes,
                          hidden_channels=hidden_channels,
                          num_layers=num_layers)
    else:
        raise ValueError

    return model.to(device)
