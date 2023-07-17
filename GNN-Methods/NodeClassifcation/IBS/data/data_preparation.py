import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Reddit2
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.utils.hetero import group_hetero_graph

def check_consistence(mode: str, batch_order: str):
    assert mode in ['ppr', 'rand', 'randfix', 'part',
                    'clustergcn', 'n_sampling', 'rw_sampling', 'ladies', 'ppr_shadow']
    if mode in ['ppr', 'part', 'randfix',]:
        assert batch_order in ['rand', 'sample', 'order']
    else:
        assert batch_order == 'rand'


def load_data(dataset_name: str,
              small_trainingset: float,
              pretransform):
    """

    :param dataset_name:
    :param small_trainingset:
    :param pretransform:
    :return:
    """
    print("dataset_name=",dataset_name)
    if dataset_name.lower() in ['arxiv', 'products', 'papers100m','mag']:
        dataset = PygNodePropPredDataset(name="ogbn-{:s}".format(dataset_name),
                                         root='./datasets',
                                         pre_transform=pretransform)
        split_idx = dataset.get_idx_split()
        graph = dataset[0]
    elif dataset_name.lower().startswith('reddit'):
        if dataset_name == 'reddit2':
            dataset = Reddit2('./datasets/reddit2', pre_transform=pretransform)
        elif dataset_name == 'reddit':
            dataset = Reddit('./datasets/reddit', pre_transform=pretransform)
        else:
            raise ValueError
        graph = dataset[0]
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        graph.train_mask, graph.val_mask, graph.test_mask = None, None, None
    else:
        raise NotImplementedError
    train_indices = split_idx["train"].numpy()

    if small_trainingset < 1:
        np.random.seed(2021)
        train_indices = np.sort(np.random.choice(train_indices,
                                                 size=int(len(train_indices) * small_trainingset),
                                                 replace=False,
                                                 p=None))

    train_indices = torch.from_numpy(train_indices)

    val_indices = split_idx["valid"]
    test_indices = split_idx["test"]
    return graph, (train_indices, val_indices, test_indices,)

def to_homo(data,split_idx):
    subject_node = list(data.y_dict.keys())[0]
    # data.node_year_dict = None
    # data.edge_reltype_dict = None
    # # remove_subject_object = ['doi']
    # # remove_pedicates = [ 'schema#awardWebpage', ]
    # to_remove_rels = []
    # to_remove_pedicates = []
    # to_remove_subject_object = []
    # to_keep_edge_idx_map = []
    # for keys, (row, col) in data.edge_index_dict.items():
    #     if (keys[2] in to_remove_subject_object) or (keys[0] in to_remove_subject_object):
    #         # print("to remove keys=",keys)
    #         to_remove_rels.append(keys)
    #
    # for keys, (row, col) in data.edge_index_dict.items():
    #     if (keys[1] in to_remove_pedicates):
    #         # print("to remove keys=",keys)
    #         to_remove_rels.append(keys)
    #         to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))
    #
    # for elem in to_remove_rels:
    #     data.edge_index_dict.pop(elem, None)
    #     data.edge_reltype.pop(elem, None)
    #
    # for key in to_remove_subject_object:
    #     data.num_nodes_dict.pop(key, None)
    ##############add inverse edges ###################
    edge_index_dict = data.edge_index_dict
    key_lst = list(edge_index_dict.keys())
    # for key in key_lst:
    #     r, c = edge_index_dict[(key[0], key[1], key[2])]
    #     edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
    ######################3
    to_remove_ind = list(set((edge_index[0] >= len(node_type)).nonzero().flatten().tolist()).union(set((edge_index[1] >= len(node_type)).nonzero().flatten().tolist())))
    if len(to_remove_ind)>0:
        to_keep_ind = [i for i in range(edge_index[0].shape[0]) if i not in to_remove_ind]
        edge_index[0]=edge_index[0][to_keep_ind]
        edge_index[1]=edge_index[1][to_keep_ind]
        edge_type = edge_type[to_keep_ind]
    ################
    homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                     node_type=node_type, local_node_idx=local_node_idx,
                     num_nodes=node_type.size(0))

    homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
    homo_data.y[local2global[subject_node]] =data.y_dict[subject_node]
    homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True
    ###########Splits ################
    # train_indicies=local2global[subject_node][split_idx["train"][subject_node]]
    # valid_indicies = local2global[subject_node][split_idx["valid"][subject_node]]
    # test_indicies = local2global[subject_node][split_idx["test"][subject_node]]

    train_indicies = split_idx["train"][subject_node]
    valid_indicies = split_idx["valid"][subject_node]
    test_indicies = split_idx["test"][subject_node]


    # train_loader = GraphSAINTRandomWalkSampler(
    #     # train_loader = GraphSAINTTaskBaisedRandomWalkSampler(
    #     # train_loader=GraphSAINTTaskWeightedRandomWalkSampler(
    #     homo_data,
    #     batch_size=args.batch_size,
    #     walk_length=args.num_layers,
    #     # Subject_indices=local2global[subject_node],
    #     # NodesWeightDic=NodesWeightDic,
    #     num_steps=args.num_steps,
    #     sample_coverage=0,
    #     save_dir=dataset.processed_dir)
    # Map informations to their canonical type.
    #######################intialize random features ###############################
    subject_node_idx=list(local2global.keys()).index(subject_node)//2
    feat = torch.Tensor(len(torch.Tensor(homo_data.node_type==subject_node_idx).nonzero()), 128)
    torch.nn.init.xavier_uniform_(feat)
    # feat_dic = {subject_node: feat}
    homo_data['x']=feat
    num_nodes_dict = {}
    print("homo_data=", homo_data)
    # for key, N in data.num_nodes_dict.items():
    #     num_nodes_dict[key2int[key]] = N
    return homo_data,train_indicies,valid_indicies,test_indicies,key2int,local2global,subject_node_idx
def load_data_hetero_homo(dataset_name: str,
              small_trainingset: float,
              pretransform):
    print("dataset_name=", dataset_name)
    dataset,split_idx=None,None
    if dataset_name.lower() in ['mag']:
        dataset = PygNodePropPredDataset(name="ogbn-{:s}".format(dataset_name),
                                         root='./datasets')
        split_idx = dataset.get_idx_split('time')
        graph,train_indices,val_indices,test_indices,key2int,local2global,subject_node_idx = to_homo(dataset[0],split_idx)
        graph = pretransform(graph)
        print("undirected graph=",graph)

    out_node = list(dataset[0].y_dict.keys())[0]
    train_indices = train_indices.numpy()
    val_indices = val_indices.numpy()
    test_indices = test_indices.numpy()
    if dataset_name.lower().startswith('reddit'):
        if dataset_name == 'reddit2':
            dataset = Reddit2('./datasets/reddit2', pre_transform=pretransform)
        elif dataset_name == 'reddit':
            dataset = Reddit('./datasets/reddit', pre_transform=pretransform)
        else:
            raise ValueError
        graph = dataset[0]
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        graph.train_mask, graph.val_mask, graph.test_mask = None, None, None

    if small_trainingset < 1:
        np.random.seed(2021)
        train_indices = np.sort(np.random.choice(train_indices,
                                                 size=int(len(train_indices) * small_trainingset),
                                                 replace=False,
                                                 p=None))


    return graph, (train_indices, val_indices, test_indices),dataset[0],key2int,split_idx,local2global,subject_node_idx

def load_data_hetero(dataset_name: str,
              small_trainingset: float,
              pretransform):
    """

    :param dataset_name:
    :param small_trainingset:
    :param pretransform:
    :return:
    """
    print("dataset_name=",dataset_name)
    if dataset_name.lower() in ['mag']:
        dataset = PygNodePropPredDataset(name="ogbn-{:s}".format(dataset_name),
                                         root='./datasets',
                                         pre_transform=pretransform)
        split_idx = dataset.get_idx_split()
        graph = dataset[0]

    out_node=list(dataset[0].y_dict.keys())[0]
    if split_idx["train"].keys():
        train_indices = split_idx["train"][out_node].numpy()

    if small_trainingset < 1:
        np.random.seed(2021)
        train_indices = np.sort(np.random.choice(train_indices,
                                                 size=int(len(train_indices) * small_trainingset),
                                                 replace=False,
                                                 p=None))

    train_indices = torch.from_numpy(train_indices)

    val_indices = split_idx["valid"][out_node]
    test_indices = split_idx["test"][out_node]
    return graph, (train_indices, val_indices, test_indices,)
class GraphPreprocess_hetero:
    def __init__(self,
                 self_loop: bool = True,
                 transform_to_undirected: bool = True):
        self.self_loop = self_loop
        self.to_undirected = transform_to_undirected

    def __call__(self, graph: Data):
        out_node=list(graph.y_dict.keys())[0]
        graph.y_dict[out_node] = graph.y_dict[out_node].reshape(-1)
        graph.y_dict[out_node] = torch.nan_to_num( graph.y_dict[out_node], nan=-1)
        graph.y_dict[out_node] =  graph.y_dict[out_node].to(torch.long)

        for key in graph.edge_index_dict.keys():
            if self.self_loop:
                graph.edge_index_dict[key], _ = add_remaining_self_loops(graph.edge_index_dict[key], num_nodes=graph.num_nodes_dict[key[0]]+graph.num_nodes_dict[key[2]])
            else:
                edge_index = graph.edge_index_dict[key]

            if self.to_undirected:
                edge_index = to_undirected(graph.edge_index_dict[key], num_nodes=graph.num_nodes_dict[key[0]]+graph.num_nodes_dict[key[2]])

            graph.edge_index_dict[key] = edge_index
        return graph


class GraphPreprocess:
    def __init__(self,
                 self_loop: bool = True,
                 transform_to_undirected: bool = True):
        self.self_loop = self_loop
        self.to_undirected = transform_to_undirected

    def __call__(self, graph: Data):
        graph.y = graph.y.reshape(-1)
        graph.y = torch.nan_to_num(graph.y, nan=-1)
        graph.y = graph.y.to(torch.long)

        if self.self_loop:
            edge_index, _ = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        else:
            edge_index = graph.edge_index

        if self.to_undirected:
            edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

        graph.edge_index = edge_index
        return graph
class GraphPreprocess_homo_hetero:
    def __init__(self,
                 self_loop: bool = True,
                 transform_to_undirected: bool = True):
        self.self_loop = self_loop
        self.to_undirected = transform_to_undirected

    def __call__(self, graph: Data):
        graph.y = graph.y.reshape(-1)
        graph.y = torch.nan_to_num(graph.y, nan=-1)
        graph.y = graph.y.to(torch.long)

        if self.to_undirected:
            edge_index_0=torch.cat((graph.edge_index[0], graph.edge_index[1]), 0)
            edge_index_1= torch.cat((graph.edge_index[1], graph.edge_index[0]), 0)
            edge_attr=torch.cat((graph.edge_attr, graph.edge_attr), 0)
            graph.edge_index=torch.stack([edge_index_0, edge_index_1])
            graph.edge_attr = edge_attr
            # edge_index, _ = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)

        if self.self_loop:
            edge_index_0 = torch.cat((graph.edge_index[0], torch.tensor(np.array(np.arange(0, graph.num_nodes)))), 0)
            edge_index_1 = torch.cat((graph.edge_index[1], torch.tensor(np.array(np.arange(0, graph.num_nodes)))), 0)
            edge_attr = torch.cat((graph.edge_attr, torch.tensor(np.array(([-1]*graph.num_nodes)))), 0)
            graph.edge_index=torch.stack([edge_index_0, edge_index_1])
            graph.edge_attr = edge_attr
            # edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)
        return graph