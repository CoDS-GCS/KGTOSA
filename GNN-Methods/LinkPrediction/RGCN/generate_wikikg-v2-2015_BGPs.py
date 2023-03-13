import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import pandas as pd
from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
from models import RGCN


if __name__ == '__main__':
    # edges_counts=(P31      2558406 instance of
    #             P106     1639330 occupation
    #             P21      1512569
    #             P735     1301107
    #             P27      1260348
    #                       ...
    #             P811           1
    #             P1773          1
    #             P1887          1
    #             P1606          1
    #             P925           1)

    # t_ds = pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/train.csv", dtype=str, header=None)
    # t_ds.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/train.txt",sep="\t",header=None,index=None)

    selected_edge="P106"

    # valid_ds=pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/valid.txt", dtype=str,sep="\t", header=None)
    # valid_ds=valid_ds.rename(columns={0:'s',1:'p',2:'o'})
    # print(valid_ds["p"].value_counts())
    # valid_ds=valid_ds[valid_ds["p"].isin([selected_edge])]
    # print(valid_ds["p"].value_counts())
    # print(valid_ds)
    # valid_ds.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/valid_"+selected_edge+".txt",header=None,index=None, sep="\t")
    # ###########################
    # test_ds = pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/test.txt",dtype=str, sep="\t", header=None)
    # test_ds = test_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    # print(test_ds["p"].value_counts())
    # test_ds = test_ds[test_ds["p"].isin([selected_edge])]
    # print(test_ds["p"].value_counts())
    # print(test_ds)
    # test_ds.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/test_"+selected_edge+".txt", header=None,
    #                 index=None, sep="\t")
    
    data_path="/shared_mnt/github_repos/RGCN_LP/data/wikikg-v2-2015/"
    train_ds = pd.read_csv(data_path+"train.txt",dtype=str, sep="\t", header=None)
    train_ds = train_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    print(train_ds)
    source_en = train_ds[train_ds["p"].isin([selected_edge])]["s"].unique().tolist()
    des_en = train_ds[train_ds["p"].isin([selected_edge])]["o"].unique().tolist()
    #########################################   
    train_SQ=train_ds[((train_ds["s"].isin(source_en)) | (train_ds["s"].isin(des_en)))]
    train_SQ=train_SQ.drop_duplicates()
    print(train_SQ)
    train_SQ.to_csv(data_path+"train"+selected_edge+"_SQ.txt", header=None,index=None, sep="\t")
    ######################################      
    train_BSQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["o"].isin(source_en)) 
                                | (train_ds["s"].isin(des_en)) | (train_ds["o"].isin(des_en)))]
    train_BSQ=train_BSQ.drop_duplicates()
    print(train_BSQ)
    train_BSQ.to_csv(data_path+"train"+selected_edge+"_BSQ.txt", header=None,index=None, sep="\t")
    # #################################
    source_source_en = train_ds[train_ds["o"].isin(source_en)]["s"].unique().tolist()
    des_des_en = train_ds[train_ds["s"].isin(des_en)]["o"].unique().tolist()    
    train_SQ_SQ = train_ds[(train_ds["s"].isin(source_source_en) | train_ds["s"].isin(des_des_en))]
    train_PQ=pd.concat([train_SQ,train_SQ_SQ])
    train_PQ=train_PQ.drop_duplicates()
    print(train_PQ)
    train_PQ.to_csv(data_path+"train"+selected_edge+"_PQ.txt", header=None,index=None, sep="\t")
    #################################
    train_q1 = train_ds[train_ds["s"].isin(source_en)|train_ds["o"].isin(source_en)]
    train_q2 = train_ds[train_ds["s"].isin(des_en)|train_ds["o"].isin(des_en)]
    train_q3 = train_ds[train_ds["s"].isin(source_source_en)|train_ds["o"].isin(source_source_en)]
    train_q4 = train_ds[train_ds["s"].isin(des_des_en)|train_ds["o"].isin(des_des_en)]
    train_BPQ=pd.concat([train_q1,train_q2,train_q3,train_q4])
    train_BPQ=train_BPQ.drop_duplicates()
    print(train_BPQ)
    train_BPQ.to_csv(data_path+"train"+selected_edge+"_BPQ.txt", header=None,index=None, sep="\t")