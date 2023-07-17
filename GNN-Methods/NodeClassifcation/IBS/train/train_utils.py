import numpy as np
import torch
import pandas as pd
from torch_sparse.tensor import SparseTensor

def run_batch(model, graph, num_microbatches_minibatch=None):
    graph.output_node_mask = graph.output_node_mask[0:len(graph.y)]
    y = graph.y[graph.output_node_mask]
    num_prime_nodes = len(y)
    outputs = model(graph)

    if hasattr(graph, 'node_norm') and graph.node_norm is not None:
        loss = torch.nn.functional.nll_loss(outputs, y, reduction='none')
        loss = (loss * graph.node_norm).sum()
    else:
        loss = torch.nn.functional.nll_loss(outputs, y)

    return_loss = loss.clone().detach() * num_prime_nodes

    if model.training:
        loss = loss / num_microbatches_minibatch
        loss.backward()

    pred = torch.argmax(outputs, dim=1)
    corrects = pred.eq(y).sum().detach()

    return return_loss, corrects, num_prime_nodes, pred, y


def homo_to_hetero(sampled_graph, org_graph):
    # dense_edge_index=sampled_graph.edge_index.to_dense()
    if type(sampled_graph.edge_index) == SparseTensor:
        row, col, edge_attr = sampled_graph.edge_index.t().coo()
        sampled_graph.edge_index  = torch.stack([row, col], dim=0)
        sampled_graph["edge_attr"] = edge_attr

    # res=[]
    # org_graph_array=org_graph.edge_index.numpy().transpose()
    # org_edge_index_df= pd.DataFrame(org_graph_array)
    # source,dest=[],[]
    # edge_attr=[]
    # for i in range(0,dense_edge_index.shape[0]):
    #     for elem in torch.nonzero(dense_edge_index[i]).flatten():
    #         # source.append(sampled_graph.nodes[i].item()) #Global node index
    #         # dest.append(sampled_graph.nodes[elem.item()].item())
    #         source.append(i) # subgraph node index
    #         dest.append(elem.item())
    #         # edge_attr.append(dense_edge_index[i][elem]//org_graph.edge_attr.max())
    #         # ,dense_edge_index[i,elem.item()].item()])
    # # res_df=pd.DataFrame(res)
    # sampled_graph.edge_index=torch.tensor([source, dest])

    sampled_graph["num_nodes"] = len(sampled_graph.nodes)
    # res_df=pd.DataFrame(list(zip(source,dest)))
    # res_df["att_idx"]=res_df.apply(lambda row: org_edge_index_df[(org_edge_index_df[0]==row[0])& (org_edge_index_df[1]==row[1])].index.values,axis=1)
    # res_df["att_idx"] = res_df["att_idx"].apply( lambda x: -1 if len(x)==0 else x[0])
    # res_df["edge_type"]=res_df["att_idx"].apply(lambda idx: -1 if idx==-1 else org_graph.edge_attr[idx])
    # sampled_graph["edge_attr"]=torch.Tensor(res_df["edge_type"].tolist())
    # sampled_graph["edge_attr"] = org_graph.edge_attr[np.arange(0,len(sampled_graph.edge_index[0]))]

    # sampled_graph["edge_attr"]=torch.Tensor(edge_attr)
    return sampled_graph


def run_rgcn_batch(model, graph, num_microbatches_minibatch=None, org_graph=None,x=None):
    # graph = homo_to_hetero(graph, org_graph)
    # graph.output_node_mask = graph.output_node_mask[0:len(graph.y)]
    # y = graph.y[graph.output_node_mask]
    # num_prime_nodes = len(y)
    outputs = model(x, graph.edge_index, graph.edge_attr, graph.node_type,graph.local_node_idx)
    outputs = outputs[graph.train_mask]
    y = graph.y[graph.train_mask].squeeze()
    loss = torch.nn.functional.nll_loss(outputs, y)
    # return_loss = loss.clone().detach() * num_prime_nodes
    if model.training:
        # loss = loss / num_microbatches_minibatch
        loss.backward()

    pred = torch.argmax(outputs, dim=1)
    corrects = pred.eq(y).sum().detach()
    acc=corrects/len(pred)

    # graph = homo_to_hetero(graph, org_graph)
    # graph.output_node_mask = graph.output_node_mask[0:len(graph.y)]
    # y = graph.y[graph.output_node_mask]
    # num_prime_nodes = len(y)
    outputs = model(x, graph.edge_index, graph.edge_attr, graph.node_type, graph.local_node_idx)
    outputs = outputs[graph.train_mask]
    y = graph.y[graph.train_mask].squeeze()
    loss = torch.nn.functional.nll_loss(outputs, y)
    # return_loss = loss.clone().detach() * num_prime_nodes
    if model.training:
        # loss = loss / num_microbatches_minibatch
        loss.backward()

    pred = torch.argmax(outputs, dim=1)
    corrects = pred.eq(y).sum().detach()
    acc = corrects / len(pred)

    return loss.clone().detach(), corrects, len(pred), pred, y,acc
