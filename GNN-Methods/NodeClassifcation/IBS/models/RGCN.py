from copy import copy
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from .evaluate import Evaluator as Evaluator
class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        # paper_count=len(x_dict[2])
        # paper_count = len(x_dict[3])
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb
            # print(key," size=",x_dict[int(key)].size())

        # print(key2int)
        # print("x_dict keys=",x_dict.keys())

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)
            # print(key,adj_t_dict[key].size)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                # print("keys=",keys)
                # print("adj_t=",adj_t)
                # print("key2int[src_key]=",key2int[src_key])
                # print("x_dict[key2int[src_key]]=",x_dict[key2int[src_key]].size())
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                # print("out size=",out.size())
                # print("tmp size=",conv.rel_lins[key2int[keys]](tmp).size())
                ################## fill missed rows hsh############################
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                out.add_(tmp)

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict

def rgcn_test(model,x_dict,data,key2int,train_nodes,val_nodes,test_nodes,subject_node='paper'):
    evaluator = Evaluator(name='ogbn-mag')
    model.eval()
    outputs = model(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)
    outputs = outputs[data.node_type == key2int[subject_node]]
    y_pred = torch.argmax(outputs, dim=1)
    y_pred=y_pred.reshape(len(y_pred), 1)
    y_true = data.y[data.node_type == key2int[subject_node]].squeeze()
    y_true=y_true.reshape(len(y_true), 1)
    print("y_true[train_nodes]=",y_true[train_nodes].shape,y_true[train_nodes])
    print("y_pred[train_nodes]=",y_pred[train_nodes].shape,y_pred[train_nodes])

    train_acc = evaluator.eval({
        'y_true': y_true[train_nodes],
        'y_pred': y_pred[train_nodes],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[val_nodes],
        'y_pred': y_pred[val_nodes],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[test_nodes],
        'y_pred': y_pred[test_nodes],
    })['acc']

    # evaluator = Evaluator(name='ogbn-mag')
    # model.eval()
    # out = model.inference(x_dict, data.edge_index_dict, key2int)
    # out = out[key2int[subject_node]]
    #
    # y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    # y_true = data.y_dict[subject_node]
    # # split_idx = data.get_idx_split('time')
    #
    # train_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['train'][subject_node]],
    #     'y_pred': y_pred[split_idx['train'][subject_node]],
    # })['acc']
    # valid_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['valid'][subject_node]],
    #     'y_pred': y_pred[split_idx['valid'][subject_node]],
    # })['acc']
    # test_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['test'][subject_node]],
    #     'y_pred': y_pred[split_idx['test'][subject_node]],
    # })['acc']
    return train_acc, valid_acc, test_acc