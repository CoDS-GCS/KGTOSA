from typing import List, Union, Tuple
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor
from data.data_utils import get_pair_wise_distance
from data.modified_tsp import tsp_heuristic

sample_graph_id=0
def getDecodedSubgraph(org_dataset, subgraph):
    global sample_graph_id
    triples_list = []
    node_types = org_dataset.node_type.unique().tolist()
    edge_types = org_dataset.edge_attr.unique().tolist()
    for idx in range(0, subgraph.edge_index.shape[1]):
        triples_list.append([
            subgraph.node_type[subgraph.edge_index[0][idx]].item(),  # Src Node Type
            subgraph.local_node_idx[subgraph.edge_index[0][idx]].item(),  # Src Node ID
            subgraph.edge_attr[idx].item(),  # relation type
            idx,
            subgraph.node_type[subgraph.edge_index[1][idx]].item(),  # Dest Node Type
            subgraph.local_node_idx[subgraph.edge_index[1][idx]].item()  # Dest Node ID
        ])
    df=pd.DataFrame(triples_list,columns=['Src_Node_Type', 'Src_Node_ID', 'Rel_type','Rel_ID', 'Dest_Node_Type','Dest_Node_ID'])
    df.to_csv("/media/hussein/UbuntuData/GithubRepos/ibmb/datasets/ibmb_ogbn_mag_subgraphs/ibmb_ogbn_mag_subgraph_"+str(sample_graph_id)+".csv",index=None)
    sample_graph_id+=1
    return df


class BaseLoader(torch.utils.data.DataLoader):
    """
    Batch-wise IBMB dataloader from paper Influence-Based Mini-Batching for Graph Neural Networks
    """

    def __init__(self,
                 dataset,
                 *args,
                 **kwargs):
        super().__init__(dataset, collate_fn=self.__collate__, **kwargs)

    def __getitem__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def loader_len(self):
        raise NotImplementedError

    def __collate__(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def normalize_adjmat(cls, adj: SparseTensor, normalization: str):
        """
        Normalize SparseTensor adjacency matrix.

        :param adj:
        :param normalization:
        :return:
        """

        assert normalization in ['sym', 'rw'], f"Unsupported normalization type {normalization}"
        assert isinstance(adj, SparseTensor), f"Expect SparseTensor type, got {type(adj)}"

        adj = adj.fill_value(1.)
        degree = adj.sum(0)

        degree[degree == 0.] = 1e-12
        deg_inv = 1 / degree

        if normalization == 'sym':
            deg_inv_sqrt = deg_inv ** 0.5
            adj = adj * deg_inv_sqrt.reshape(1, -1)
            adj = adj * deg_inv_sqrt.reshape(-1, 1)
        elif normalization == 'rw':
            adj = adj * deg_inv.reshape(-1, 1)

        return adj

    @classmethod
    def indices_complete_check(cls,
                               loader: List[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]],
                               output_indices: Union[torch.Tensor, np.ndarray]):
        if isinstance(output_indices, torch.Tensor):
            output_indices = output_indices.cpu().numpy()

        outs = []
        for out, aux in loader:
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            if isinstance(aux, torch.Tensor):
                aux = aux.cpu().numpy()

            assert np.all(np.in1d(out, aux)), "Not all output nodes are in aux nodes!"
            outs.append(out)

        outs = np.sort(np.concatenate(outs))
        assert np.all(outs == np.sort(output_indices)), "Output nodes missing or duplicate!"

    @classmethod
    def get_subgraph(cls,
                     out_indices: torch.Tensor,
                     graph: Data,
                     return_edge_index_type: str,
                     adj: SparseTensor,FG_adj_df=None,
                     **kwargs):
        if return_edge_index_type == 'adj':
            assert adj is not None

        if return_edge_index_type == 'adj':
            if (FG_adj_df is not None) &  ('node_type' in graph.to_dict().keys()):
                edge_index_list = (np.in1d(FG_adj_df[0].values,out_indices) & np.in1d(FG_adj_df[1].values,out_indices)).nonzero()[0]
                g_adj_df=FG_adj_df.loc[edge_index_list]
                nodes_ids=list(sorted(set(g_adj_df[0].values.tolist()).union(set(g_adj_df[1].values.tolist()))))
                nodes_idx=list(range(0,len(nodes_ids)))
                nodes_dict=dict(zip(nodes_ids, nodes_idx))
                g_adj_df[0]=g_adj_df[0].apply(lambda x: nodes_dict[x])
                g_adj_df[1] = g_adj_df[1].apply(lambda x: nodes_dict[x])
                # g_adj_df = FG_adj_df[FG_adj_df[0].isin(out_indices.numpy()) | FG_adj_df[1].isin(out_indices.numpy())]

                ##############################
                edge_index = torch.Tensor(g_adj_df.iloc[:, 0:2].to_numpy().transpose()).long()
                edge_attr = torch.Tensor(g_adj_df.iloc[:, 2].to_numpy().transpose()).long()
                num_nodes = len(out_indices)
            else:
                edge_index = adj[out_indices, :][:, out_indices]
                edge_attr = graph.edge_attr[adj[out_indices, :][:, out_indices].nonzero()]

            if 'node_type' in graph.to_dict().keys():
                subg = Data(
                    y=graph.y[out_indices], edge_attr=edge_attr,
                    edge_index=edge_index, nodes=out_indices,
                    num_nodes=num_nodes, node_type=graph.node_type[out_indices],
                    local_node_idx=graph.local_node_idx[out_indices],
                    train_mask=graph.train_mask[out_indices], return_edge_index_type='adj')
                # decoded_subgraph=getDecodedSubgraph(graph,subg)
            else:
                subg = Data(x=graph.x[out_indices],
                            y=graph.y[out_indices],
                            edge_index=adj[out_indices, :][:, out_indices])

        elif return_edge_index_type == 'edge_index':
            edge_index, edge_attr = subgraph(out_indices,
                                             graph.edge_index,
                                             graph.edge_attr,
                                             relabel_nodes=True,
                                             num_nodes=graph.num_nodes,
                                             return_edge_mask=False)
            subg = Data(x=graph.x[out_indices],
                        y=graph.y[out_indices],
                        edge_index=edge_index,
                        edge_attr=edge_attr, return_edge_index_type='edge_index')
        else:
            raise NotImplementedError

        for k, v in kwargs.items():
            subg[k] = v

        return subg

    @classmethod
    def get_subgraph_hetero(cls,
                            out_indices: torch.Tensor,
                            graph: Data,
                            return_edge_index_type: str,
                            adj: SparseTensor,
                            **kwargs):
        out_node_type = list(graph.x_dict.keys())[0]
        if return_edge_index_type == 'adj':
            assert adj is not None

        if return_edge_index_type == 'adj':
            out_indices = out_indices[out_indices < len(graph.y_dict[out_node_type])]  ## ToDo filter only output nodes
            subg = Data(x=graph.x_dict[out_node_type][out_indices],
                        y=graph.y_dict[out_node_type][out_indices],
                        edge_index=adj[out_indices, :][:, out_indices])

        elif return_edge_index_type == 'edge_index':
            edge_index, edge_attr = subgraph(out_indices,
                                             graph.edge_index,
                                             graph.edge_attr,
                                             relabel_nodes=True,
                                             num_nodes=graph.num_nodes,
                                             return_edge_mask=False)
            subg = Data(x=graph.x[out_indices],
                        y=graph.y[out_indices],
                        edge_index=edge_index,
                        edge_attr=edge_attr)
        else:
            raise NotImplementedError

        for k, v in kwargs.items():
            subg[k] = v

        return subg

    @classmethod
    def define_sampler(cls,
                       batch_order: str,
                       ys: List[Union[torch.Tensor, np.ndarray, List]],
                       num_classes: int,
                       dist_type: str = 'kl'):
        if batch_order == 'rand':
            logging.info("Running with random order")
            sampler = RandomSampler(ys)
        elif batch_order in ['order', 'sample']:
            kl_div = get_pair_wise_distance(ys, num_classes, dist_type=dist_type)
            if batch_order == 'order':
                best_perm, _ = tsp_heuristic(kl_div)
                logging.info(f"Running with given order: {best_perm}")
                sampler = OrderedSampler(best_perm)
            else:
                logging.info("Running with weighted sampling")
                sampler = ConsecutiveSampler(kl_div)
        else:
            raise ValueError

        return sampler
