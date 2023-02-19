import os

import pandas as pd
import numpy as np
import re

if __name__ == '__main__':
    ogb_datset_path = "/home/hussein/Downloads/mag/mapping"
    relidx2relnameFile = "relidx2relname.csv"
    labelidx2lblnameFile = "labelidx2venuename.csv"
    index_file_patten = '_entidx2name'
    ent_idx = "ent idx"
    ent_name = "ent name"
    namespace = "http://mag.graph/"
    triples_lst = []
    obg_ds_vertices_types = []
    obg_ds_vertices_dfs = {}
    obg_ds_idx_relations_dic = {}
    ###################load vertices####################
    directory = os.fsencode(ogb_datset_path)
    papers_df = None
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(index_file_patten + ".csv"):
            vertix_type = filename.split(index_file_patten)[0]
            print("vertix_type=", vertix_type)
            obg_ds_vertices_types.append(vertix_type)
            temp_df = pd.read_csv(ogb_datset_path + "/" + filename, sep=",")
            if vertix_type in ['paper']:
                papers_df = temp_df.copy()
                paper_year = pd.read_csv(ogb_datset_path + "/node-feat/paper/node_year.csv", sep=",", header=None)
                print("paper_year=", paper_year.head())
                papers_df["year"] = paper_year[0]
                venues = pd.read_csv(ogb_datset_path + "/labelidx2venuename.csv", sep=",")
                paper_venue = pd.read_csv(ogb_datset_path + "/node-label/paper/node-label.csv", sep=",", header=None)
                papers_df["venue_id"] = paper_venue[0]
                papers_df.to_csv("obgn_mag_paper.csv", sep=",", index=None)
            obg_ds_vertices_dfs[vertix_type] = pd.Series(temp_df[ent_name].values, index=temp_df[ent_idx]).to_dict()
            # print(obg_ds_vertices_dfs[vertix_type].head())

            continue
        else:
            continue
    relidx2relname_df = pd.read_csv(ogb_datset_path + "/" + relidx2relnameFile, sep=",")
    for idx, row in relidx2relname_df.iterrows():
        obg_ds_idx_relations_dic[row[0]] = row[1]
    print("obg_ds_idx_relations_dic=", obg_ds_idx_relations_dic)
    # print("relidx2relname_df=", relidx2relname_df.head())
    labelidx2lblname_df = pd.read_csv(ogb_datset_path + "/" + labelidx2lblnameFile, sep=",")
    # print("labelidx2lblname_df=", labelidx2lblname_df.head())

    ###########################map triples #####################
    df_parts = []
    for root, subdirectories, files in os.walk(ogb_datset_path + "/relations/"):
        # for subdirectory in subdirectories:
        #    print(os.path.join(root, subdirectory))
        for file in files:
            # filename = os.fsdecode(file)
            if file in ["edge.csv"]:
                edge_df = pd.read_csv(root + "/" + file, sep=",", header=None)
                edge_df = edge_df.rename(columns={0: "s", 1: "o"})
                edge_reltype_df = pd.read_csv(root + "/" + "edge_reltype.csv", sep=",", header=None)
                edge_reltype_df = edge_reltype_df.rename(columns={0: "p"})
                edge_df["p"] = edge_reltype_df["p"]
                edge_df["p"] = edge_df["p"].apply(lambda x: namespace + obg_ds_idx_relations_dic[x])
                edge_reltype_df = None
                num_edge_list_df = pd.read_csv(root + "/" + "num-edge-list.csv", sep=",", header=None)
                if len(edge_df) == int(num_edge_list_df[0][0]):
                    triple = root.split("/")[-1].split("___")  # get last element
                    h_dic = obg_ds_vertices_dfs[triple[0]]
                    edge_df["s"] = edge_df["s"].apply(lambda x: namespace + triple[0] + "/" + str(h_dic[int(x)]))
                    t_dic = obg_ds_vertices_dfs[triple[2]]
                    edge_df["o"] = edge_df["o"].apply(lambda x: namespace + triple[2] + "/" + str(t_dic[int(x)]))
                    r = triple[1]
                    # if r in ['writes']:
                    #    edge_df_inverse=edge_df.copy()
                    #    edge_df_inverse=edge_df_inverse.rename(columns={"s": "o1", "o": "s"})
                    #    edge_df_inverse=edge_df_inverse.rename(columns={"o1":"o"})
                    #    edge_df_inverse = edge_df_inverse.reindex(columns=['s', 'o', 'p'])
                    #    edge_df_inverse["p"]=edge_df_inverse["p"].apply(lambda x: str(x).replace("/writes","/writtenby"))
                    #    edge_df=pd.concat([edge_df,edge_df_inverse])
                    edge_df.to_csv(root.split("/")[-1] + ".csv", sep="\t", header=None)
                    # print("len=true")

                print("file=", root + "/" + file)
                print("len edge_df", len(edge_df))
                print("edge_df=", edge_df.head())
                df_parts.append(edge_df)
            # vertix_type = filename.split(index_file_patten)[0]
            # print("filename=", filename)
            # obg_ds_vertices_types.append(vertix_type)
            # obg_ds_vertices_dfs[vertix_type] = pd.read_csv(ogb_datset_path + "/" + filename, sep=",")
            # print(obg_ds_vertices_dfs[vertix_type].head())
            continue
        else:
            continue
    ####################add label ##################33
    paper_venue_df = papers_df.drop(columns=["ent idx", "year"], axis=1)
    paper_venue_df = paper_venue_df.rename(columns={"ent name": "s", "venue_id": "o"})
    paper_venue_df["s"] = paper_venue_df["s"].apply(lambda x: namespace + "paper/" + str(x))
    paper_venue_df["p"] = namespace + "has_venue"
    df_parts.append(paper_venue_df)
    ####################add year ##################33
    paper_year_df = papers_df.drop(columns=["ent idx", "venue_id"], axis=1)
    paper_year_df = paper_year_df.rename(columns={"ent name": "s", "year": "o"})
    paper_year_df["s"] = paper_year_df["s"].apply(lambda x: namespace + "paper/" + str(x))
    paper_year_df["p"] = namespace + "has_year"
    df_parts.append(paper_year_df)

    final_df = pd.concat(df_parts)
    final_df = final_df[["s", "p", "o"]]
    final_df.to_csv("ogb-mag.tsv", sep="\t", index=None)

    # print("obg_ds_vertices_types=",obg_ds_vertices_types)
    # obg_ds_vertices_types=['author','field_of_study','institution','labelidx2venuename.csv','paper_entidx2name.csv']
    # obg_ds_rel_idx_file ='relidx2relname.csv'
    # relations_mapping={}
    # pd.read_csv("")
