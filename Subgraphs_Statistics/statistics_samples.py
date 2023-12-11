import pandas as pd #for handling csv and csv contents
from multiprocessing import Pool
from functools import partial
from tqdm_multiprocess import TqdmMultiProcessPool
import tqdm
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
from networkx import Graph as NXGraph
from networkx import NetworkXNoPath
import matplotlib.pyplot as plt
import statistics
import collections
from threading import Thread
import scipy.sparse
import scipy.sparse.csgraph
import numpy as np
# from rdflib.namespace import FOAF , XSD #most common namespaces
# import urllib.parse #for parsing strings to URI's
import os.path
max_d = []
node_types_per_hop= {}
edge_types_per_hop= {}

def get_node_type(node= None):
    if node is None : 
        return None
    else: 
        # print("the type of this node is ", str(node).split("/")[3] )
        return str(node).split("/")[3]
    
# edge type of type inverse should not be considered
def get_edge_type(edge= None):
    if edge is None or "inv" in str(edge["triples"][0][1]).split("/")[3]: 
        return None
    else: 
        # print("the type of this edge is ", str(edge["triples"][0][1]).split("/")[3] )
        return str(edge["triples"][0][1]).split("/")[3]

    
    
def get_dfs_depth(G, source=None, depth_limit=None):
    if source is None:
        nodes = G
    else:
        nodes = source
    visited = set()

    if depth_limit is None:
        depth_limit = len(G)
        print("limit is : ", depth_limit)
    max_depth = 0
    for start in nodes:
        if start in visited:
            continue
       
        # visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]
        visited_per_src = set()
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                
                if child not in visited_per_src:
                    # yield parent, child
                    # visited.add(child)
                    visited_per_src.add(child)
                    if depth_now > 1:
                        if (depth_limit - depth_now + 1)>max_depth :
                            max_depth = depth_limit - depth_now + 1
                        stack.append((child, depth_now - 1, iter(G[child])))
            except StopIteration:
                stack.pop()
    return max_depth


def stat_per_hop(G, source=None, hop_limit=None):
    global node_types_per_hop
    global edge_types_per_hop
    visited = set()
    if source is None:
        nodes = G
    else:
        nodes = source
        
    if hop_limit is None:
        hop_limit = len(G)
    
    
        
    for start in nodes:
        
        # if start in visited:
        #     print("already visited", start)
        #     continue
            
        node_types_per_hop[start] = {}
        edge_types_per_hop[start] =  {}
        for i in range (0,hop_limit+1):
            node_types_per_hop[start][i] =  set()
            edge_types_per_hop[start][i] =  set()
        
#         should not be added to node type because it is not hop 0 nodes but root nodes 
        # node_types[0].add(getType(child))
        # visited.add(start)
        queue = [(start, 0, iter(G[start]))]
        index = 0
        while len(queue) > index:
            parent, hop_now, children = queue[index]
            for child in children:
                if child not in visited:
                    yield parent, child
                    # print("node is ", child)
                    # visited.add(child)
                    # make sure to get the type of child node
                    # also how to get the edge that connected parent to child ?
                    edge_types_per_hop[start][hop_now].add(get_edge_type(G.get_edge_data(parent,child)))
                    node_types_per_hop[start][hop_now].add(get_node_type(child))
                    if hop_now + 1 <= hop_limit: 
                        queue.append((child, hop_now + 1, iter(G[child])))
            index = index + 1
    return node_types_per_hop
    
def immediate_neigh(G, source=None):
    count_immediate_neighbor = 0
    min=len(G.nodes)
    max=0
    if source is None:
        nodes = G
    else:
        nodes = source
        
    for start in nodes:
        count_immediate_neighbor = count_immediate_neighbor + len(G[start])
        min=len(G[start]) if len(G[start])<min else min
        max = len(G[start]) if len(G[start]) > max else max
    return min,max,count_immediate_neighbor/len(nodes)

def avg_shared_neigh(G, source=None,target_node=None):
    count_shared_neighbor = 0
    if source is None:
        nodes = G
    else:
        nodes = source
        
    for start in nodes:
        neighbors = G[start]
        for neighbor in neighbors: 
            next_neighbors =  G[neighbor]
            for next_neighbor in next_neighbors:
                if get_node_type(next_neighbor) == target_node:
                    # print("shared node") 
                    count_shared_neighbor = count_shared_neighbor + 1
                    break

    return count_shared_neighbor * 1.0 /len(nodes)
       
    
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def number_of_pendants(g):
    """
    Equals the number of nodes with degree 1
    """
    pendants = 0
    for u in g:
        if g.degree[u] == 1:
            pendants += 1
    return pendants

def histogram(l):
    degree_sequence = sorted([d for n, d in list(l.items())], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    print(deg, cnt)

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Histogram")
    plt.ylabel("Count")
    plt.xlabel("Value")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()
 # problem with this code is that it does not count the distinct nodes but triplese
def NodesAndEdgeTypesHistogram(df,n_nodes,n_edges):
    dic_nodes_types={}
    for nodeType in df["Src_Node_Type"].unique():
        dic_nodes_types[nodeType]=len(df[df["Src_Node_Type"].isin([nodeType])]["Src_Node_ID"].unique())
    dic_nodes_types={k: (v,v/n_nodes) for k, v in sorted(dic_nodes_types.items(), key=lambda item: item[1],reverse=True)}
    dic_edge_types = {}
    # we have inverse edges which should not be considered as a new type
    for edgeType in df["Rel_type"].unique():
        if "inv" not in edgeType:
            dic_edge_types[edgeType] = len(df[df["Rel_type"].isin([edgeType])])
    dic_edge_types = {k: (v,v/n_edges) for k, v in sorted(dic_edge_types.items(), key=lambda item: item[1],reverse=True)}
    return {"nodes":dic_nodes_types,"edges":dic_edge_types}
def HistogramPPR(df):
    dic_nodes_types_ppr_score_dic={}
    dest_type=df["Dest_Node_Type"].unique().tolist()
    for nodeType in dest_type:
        type_df=df[df["Dest_Node_Type"]==nodeType]
        dic_nodes_types_ppr_score_dic[nodeType]=(type_df["ppr_score"].min(),type_df["ppr_score"].max(),type_df["ppr_score"].mean(),type_df["ppr_score"].std())
    dic_nodes_types_ppr_score_dic = {k: v for k, v in sorted(dic_nodes_types_ppr_score_dic.items(), key=lambda item: item[1][2], reverse=True)}
    return dic_nodes_types_ppr_score_dic

filtered_dict={}
def get_closeness_centrality(nx_g,target_nodes):
    for node in target_nodes:
        filtered_dict[node] = nx.closeness_centrality(nx_g, u=node)
        # print("len(filtered_dict)=",len(filtered_dict))
    return filtered_dict

def construct_nx_graph(data_df):
    rdf_g = Graph()
    prefix_str = 'http://yago-knowledge.org/'
    prefix = Namespace(prefix_str)
    rdf_g.bind('yago_prefix', prefix)
    S_URI = P_URI = O_URI = None
    for index, row in data_df.iterrows():
        S_URI = URIRef(prefix_str + str(row[0]) + "/" + str(row[1]))
        P_URI = URIRef(prefix_str + str(row[2]))
        O_URI = URIRef(prefix_str + str(row[3]) + "/" + str(row[4]))
        rdf_g.add((S_URI, P_URI, O_URI))
    nx_g = rdflib_to_networkx_graph(rdf_g)
    return nx_g
def node_degree_per_type(nx_g):
    dic_deg={}
    dic_count={}
    for (uri,deg) in list(nx_g.degree):
        type=str(uri).split("/")[-2]
        if type not in dic_deg:
            dic_deg[type] = 0
            dic_count[type]=0
        dic_deg[type]+=int(deg)
        dic_count[type]+= 1
    dic_deg = {k:(v,dic_count[k],v/dic_count[k]) for k, v in dic_deg.items()}
    dic_deg = {k: v for k, v in sorted(dic_deg.items(), key=lambda item: item[1][2], reverse=True)}
    return dic_deg

def getPPR_scors_dict(path):
    df=pd.read_csv(path)
    scores_dict=dict(zip(list(zip(df["src"].tolist(), df["dest"].tolist())),df["ppr_score"].tolist()))
    return scores_dict

def calc_single_source_shortest_path_length(nx_g,target_nodes,src):
    dic_length = nx.single_source_shortest_path_length(nx_g, src,cutoff=5)
    distances=[v for k, v in dic_length.items() if k in target_nodes and v>0] # avergae from connected targets only
    maxdistance = avg_dist = 0
    if len(distances) > 0:
        maxdistance = max(distances)
        avg_dist = mean(distances)
    return maxdistance, avg_dist
def AVG_NODE_DISTANCE_from_TARGET_Parallel(nx_g,target_nodes,non_target_nodes):
    # print("AVG NODE DISTANCE from TARGET")
    num_disconnected = 0
    maxdistance_lst=[]
    avg_dist_lst=[]
    print("start calc_single_source_shortest_path_length")
    n_cores=12
    non_target_nodes=list(non_target_nodes)
    for elem in tqdm.tqdm(range(0,len(non_target_nodes)//n_cores)):
        with Pool(n_cores) as p:
            result= list(p.map(partial(calc_single_source_shortest_path_length, nx_g,target_nodes),non_target_nodes[elem*n_cores:elem*n_cores+n_cores]))
        result=list(zip(*result))
        num_disconnected+=len([elem for elem in list(result[0]) if elem ==0])
        maxdistance_lst.append(max(list(result[0])))
        avg_dist_lst.append(mean(list(result[1])))
    return max(maxdistance_lst),mean(avg_dist_lst),num_disconnected, (len(non_target_nodes) - num_disconnected)

def AVG_NODE_DISTANCE_from_TARGET(nx_g,target_nodes,non_target_nodes):
    # print("AVG NODE DISTANCE from TARGET")
    num_disconnected = 0
    maxdistance_lst=[]
    avg_dist_lst=[]
    print("start calc_single_source_shortest_path_length")
    n_cores=12
    non_target_nodes=list(non_target_nodes)
    for elem in tqdm.tqdm(non_target_nodes):
        pmax,pavg=calc_single_source_shortest_path_length(nx_g,target_nodes,elem)
        if pmax==0:
            num_disconnected+=1
        if pavg>0:
            maxdistance_lst.append(pmax)
            avg_dist_lst.append(pavg)
    return max(maxdistance_lst),mean(avg_dist_lst),num_disconnected, (len(non_target_nodes) - num_disconnected)

def AVG_NODE_DISTANCE_from_TARGET_include_disconnected(nx_g,target_nodes,non_target_nodes):
    # print("AVG NODE DISTANCE from TARGET")
    num_disconnected = 0
    maxdistance_lst=[]
    avg_dist_lst=[]
    print("start calc_single_source_shortest_path_length")
    n_cores=12
    non_target_nodes=list(non_target_nodes)
    for elem in tqdm.tqdm(non_target_nodes):
        pmax,pavg=calc_single_source_shortest_path_length(nx_g,target_nodes,elem)
        if pmax==0:
            num_disconnected+=1
        maxdistance_lst.append(pmax)
        avg_dist_lst.append(pavg)
    return max(maxdistance_lst),mean(avg_dist_lst),num_disconnected, (len(non_target_nodes) - num_disconnected)

if __name__ == '__main__':
    datasets={
              "YAGO_WG_Samples_20000TN":{"target_node":"CreativeWork","target_node_type_id":1,"samplers":["ibmb_YAGO30M_CW_subgraphs_20000PN","YAGO_CW_URW_GS_DecodedSubgraph_20000TN","YAGO_CW_BRW_GS_DecodedSubgraph_20000TN","YAGO_CW_d1h1_GS_DecodedSubgraph_20000TN"],'ibmb_scores_path':'ibmb_YAGO30M_CW_ppr_Scores.csv'},
              "OGBN-MAG_PV_Samples_20000TN":{"target_node":"Paper","target_node_type_id":64,"samplers":["ibmb_mag42M_subgraph_20000PN","OGBN-MAG_PV_BRW","OGBN-MAG_PV_BRW_d1h1","OGBN-MAG_PV_URW","OGBN-MAG_PV_URW_d1h1"],'ibmb_scores_path':'ibmb_MAG42M_ppr_Scores.csv'},
              "DBLP_PV_Samples_20000TN":{"target_node":"rec","target_node_type_id":38,"samplers":["ibmb_DBLP15M_PV_subgraphs_20000PN","DBLP_SQ_GS_DecodedSubgraph","DBLP_FG_DecodedSubgraph","DBLP_FG_BiasedGS_DecodedSubgraph"],'ibmb_scores_path':'ibmb_DBLP15M_PV_ppr_Scores.csv'}
            }
    for k,v in datasets.items():
        Samplers=v["samplers"]
        for sampler in Samplers:
            target_node = v["target_node"]
            # target_nodes = [64, 38, 1]  # [Paper,rec,CreativeWork]
            target_node_id = v["target_node_type_id"]
            res = []
            target_node=str(target_node_id) if sampler.startswith("ibmb_") else target_node
            ppr_dict={}
            if sampler.startswith("ibmb_"):
                ppr_dict=getPPR_scors_dict("/media/hussein/UbuntuData/GithubRepos/ogb_cods/examples/nodeproppred/mag/Sampled_Subgraphs/"+k+"/"+v["ibmb_scores_path"])
            for idx in range(0,3):
                res.append({})
                print("sample idx=",sampler,idx)
                # res[idx]={}
                file_path="/media/hussein/UbuntuData/GithubRepos/ogb_cods/examples/nodeproppred/mag/Sampled_Subgraphs/"+k+"/"+sampler+"/"+sampler+"_"+str(idx+1) + ".csv"
                if os.path.isfile(file_path):
                    res[idx]['name']=sampler+"_"+ str(idx+1) + ".csv"
                    res[idx]["target_node"]=target_node
                    # print("######################################################################################################")
                    print(sampler+ "_"+str(idx+1) + ".csv")
                    data_df = pd.read_csv(file_path)
                    data_df=data_df[['Src_Node_Type','Src_Node_ID','Rel_type','Dest_Node_Type','Dest_Node_ID']]
                    for col in data_df.columns:
                        data_df[col] = data_df[col].astype(str)
                    # data_df = data_df.rename(columns={0: "Src_Node_Type", 1: "Src_Node_ID", 2: "Rel_Type", 3:"Rel_ID" , 4: "Dest_Node_Type", 5:"Dest_Node_ID"}, errors="raise")

                    nx_g=construct_nx_graph(data_df)
                    num_nodes=len(nx_g.nodes)
                    num_edges =len(nx_g.edges)
                    dic_type_stat = NodesAndEdgeTypesHistogram(data_df,num_nodes,num_edges)
                    dic_node_type = dic_type_stat["nodes"]
                    # print("dic_node_type=",dic_node_type)
                    res[idx]['dic_node_type'] =dic_node_type
                    dic_edge_type = dic_type_stat["edges"]
                    # print("dic_edge_type=", dic_edge_type)
                    res[idx]['dic_edge_type'] = dic_edge_type
                    node_type_deg_dic=node_degree_per_type(nx_g)
                    # print("node_type_deg_dic",node_type_deg_dic)
                    res[idx]['node_type_deg_dic'] = node_type_deg_dic
                    num_node_types = len(dic_node_type)
                    num_edge_types = len(dic_edge_type)
                    num_target_nodes = dic_node_type[target_node]
                    # print("# NODE TYPE ")
                    # print("============")
                    res[idx]['num_node_types']=num_node_types
                    # print("# EDGE TYPE ")
                    # print("============")
                    # print(num_edge_types)
                    res[idx]['num_edge_types'] = num_edge_types
                    print("# num_target_nodes")
                    print("============")
                    print(num_target_nodes)
                    print("# Graph Density ")
                    print("============")
                    print(nx.density(nx_g))
                    res[idx]['num_nodes'] = num_nodes
                    res[idx]['num_edges'] = num_edges
                    res[idx]['graph_density'] = nx.density(nx_g)
                    res[idx]['num_target_nodes_dic'] = num_target_nodes
                    target_nodes = {k for k in nx_g.nodes() if target_node in k}
                    res[idx]['target_nodes_count'] = len(target_nodes)
                    res[idx]['target_nodes_ratio'] = len(target_nodes) / num_nodes
                    non_target_nodes = {k for k in nx_g.nodes() if target_node not in k}
                    #####################
                    max_target_dist, avg_target_dist, num_disconnected, num_connected=AVG_NODE_DISTANCE_from_TARGET(nx_g, target_nodes,non_target_nodes)
                    res[idx]['avg_target_dist']=avg_target_dist
                    res[idx]['max_target_dist'] = max_target_dist
                    res[idx]['num_target_disconnected'] = num_disconnected
                    res[idx]['num_target_connected'] = num_connected
                    res[idx]['disconnected_ratio'] = num_disconnected / num_nodes
                    #####################
                    print("# AVG Target NEIGHBORS NODES")
                    print("============")
                    min,max,avg=immediate_neigh(nx_g,target_nodes)
                    res[idx]['target_neighbours_min'] = min
                    res[idx]['target_neighbours_max'] = max
                    res[idx]['target_neighbours_avg'] = avg
                    # print(min,max,avg)
                    # print("# AVG SHARED NEIGHBORS")
                    # print("============")
                    avg_shared=avg_shared_neigh(nx_g,target_nodes,target_node)
                    # print()
                    res[idx]['target_shared_neighbours_avg'] = avg_shared
                    if sampler.startswith("ibmb_"):
                        data_df["ppr_score"] = data_df.apply(lambda row: ppr_dict[(int(row.Src_Node_ID), int(row.Dest_Node_ID))] if (int(row.Src_Node_ID),int(row.Dest_Node_ID)) in ppr_dict.keys() else 0,axis=1)
                        ppr_dict=HistogramPPR(data_df)
                        res[idx]['ppr_scores'] = ppr_dict
                    print("res[idx]=",res[idx])

            # print(res)
            df = pd.DataFrame(res)
            df.to_csv(sampler+".tsv",sep="\t",index=None)