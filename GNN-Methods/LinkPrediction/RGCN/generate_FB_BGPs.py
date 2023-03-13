import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import pandas as pd
from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
from models import RGCN


if __name__ == '__main__':

    # valid_ds=pd.read_csv("/data/FB15k-237/valid_original.txt", sep="\t", header=None)
    # valid_ds=valid_ds.rename(columns={0:'s',1:'p',2:'o'})
    # print(valid_ds["p"].value_counts())
    # valid_ds=valid_ds[valid_ds["p"].isin(["/people/person/profession"])]
    # print(valid_ds["p"].value_counts())
    # print(valid_ds)
    # valid_ds.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/valid.txt",header=None,index=None, sep="\t")
    # ###########################
    # test_ds = pd.read_csv("/data/FB15k-237/test_original.txt", sep="\t", header=None)
    # test_ds = test_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    # print(test_ds["p"].value_counts())
    # test_ds = test_ds[test_ds["p"].isin(["/people/person/profession"])]
    # print(test_ds["p"].value_counts())
    # print(test_ds)
    # test_ds.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/test.txt", header=None,
    #                 index=None, sep="\t")
    #########################################
    # train_ds = pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train.txt", sep="\t", header=None)
    # print(train_ds)
    # train_ds = train_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    # source_en=train_ds[train_ds["p"].isin(["/people/person/profession"])]["s"].unique().tolist()
    # des_en = train_ds[train_ds["p"].isin(["/people/person/profession"])]["o"].unique().tolist()
    # # train_prof_SQ=train_ds[((train_ds["s"].isin(source_en)) | (train_ds["o"].isin(des_en)))]
    # # print(train_prof_SQ)
    # # train_prof_SQ.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train_prof_SQ.txt", header=None,
    # #                index=None, sep="\t")
    #
    # train_prof_BSQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["o"].isin(source_en)) | (train_ds["o"].isin(des_en)) | (
    #         train_ds["s"].isin(des_en)))]
    # print(train_prof_BSQ)
    # train_prof_BSQ.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train_prof_BSQ.txt", header=None,
    #                      index=None, sep="\t")
    #################################
    # train_ds = pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train.txt", sep="\t", header=None)
    # print(train_ds)
    # train_ds = train_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    # source_en = train_ds[train_ds["p"].isin(["/people/person/profession"])]["s"].unique().tolist()
    # source_source_en = train_ds[train_ds["o"].isin(source_en)]["s"].unique().tolist()
    # # train_prof_SQ=train_ds[((train_ds["s"].isin(source_en)) | (train_ds["o"].isin(des_en)))]
    # # print(train_prof_SQ)
    # # train_prof_SQ.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train_prof_SQ.txt", header=None,
    # #                index=None, sep="\t")
    #
    # train_prof_PQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["o"].isin(source_en) & train_ds["s"].isin(source_source_en)))]
    # print(train_prof_PQ)
    # train_prof_PQ.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train_prof_PQ.txt", header=None,
    #                       index=None, sep="\t")
    #################################
    train_ds = pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train.txt", sep="\t", header=None)
    print(train_ds)
    train_ds = train_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    source_en = train_ds[train_ds["p"].isin(["/people/person/profession"])]["s"].unique().tolist()
    dest_en = train_ds[train_ds["p"].isin(["/people/person/profession"])]["o"].unique().tolist()
    source_source_en = train_ds[train_ds["o"].isin(source_en)]["s"].unique().tolist()
    dest_dest_en = train_ds[train_ds["s"].isin(dest_en)]["o"].unique().tolist()
    train_prof_BPQ = train_ds[
        ((train_ds["s"].isin(source_en)) |(train_ds["o"].isin(dest_en))
         |(train_ds["o"].isin(source_en) & train_ds["s"].isin(source_source_en))
         |(train_ds["s"].isin(dest_en) & train_ds["o"].isin(dest_dest_en)))]
    print(train_prof_BPQ)
    train_prof_BPQ.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/FB15k-237/train_prof_BPQ.txt", header=None,
                         index=None, sep="\t")