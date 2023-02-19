import pickle
from more_itertools import sliced
import random as random 
import gc
import numpy
import numpy as np
import  pandas as pd
import gzip
import os
import datetime
import shutil
import multiprocessing
from joblib import Parallel, delayed
import torch
from tqdm import tqdm
import multiprocessing
manager = multiprocessing.Manager()
distinct_source_nodes_set=set(list(range(1,100000)))
n_negative_samples=1000
test_negatives_links_lst=shared_list = manager.list()
valid_negatives_links_lst=shared_list = manager.list()

def addNegSamples(distinct_source_nodes_set,test_valid_dic,lst_keys,test_negatives_links_lst,valid_negatives_links_lst,n_negative_samples):
    for key in lst_keys:
        neg_list = list(distinct_source_nodes_set - set(test_valid_dic[key]))
        # rand_test_idx = random.randint(0, len(neg_list) - n_negative_samples)
        test_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))
        # test_negatives_links_lst.append(neg_list[rand_test_idx:rand_test_idx+n_negative_samples])
        # rand_valid_idx = random.randint(0, len(neg_list) - n_negative_samples)
        valid_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))

from sklearn.metrics import precision_recall_fscore_support as score
def compress_gz(f_path,delete_file=True):
    f_in = open(f_path,'rb')
    f_out = gzip.open(f_path+".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    if delete_file:
        os.remove(f_path)


test_negatives_links_lst=[]
valid_negatives_links_lst = []
distinct_source_nodes_set=[]
test_valid_dic={}
def addNegativeSamples(keys,n_negative_samples):
    for key in keys:
        neg_list = list(distinct_source_nodes_set - set(test_valid_dic[key]))
        rand_test_idx = random.randint(0, len(neg_list) - n_negative_samples)
        test_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))
        # test_negatives_links_lst.append(neg_list[rand_test_idx:rand_test_idx+n_negative_samples])
        # rand_valid_idx = random.randint(0, len(neg_list) - n_negative_samples)
        valid_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))
###################### Zip Folder to OGB Format
#zip -r mag_ComputerProgramming_papers_venue_QM3.zip mag_ComputerProgramming_papers_venue_QM3/ -i '*.gz'

if __name__ == '__main__':
    split_rel="https://dblp.org/rdf/schema#yearOfPublication"
    pred_link_predicate = "https://dblp.org/rdf/schema#authoredBy"
    # target_rel = "http://mag.graph/has_venue"
    # title_rel="http://mag.graph/title"
    remove_predicates=[]
    # fieldOfStudy_Coverage_df = pd.read_csv("/media/hussein/UbuntuData/OGBN_Datasets/Sparql_Sampling/ogbn_mag_fieldOfStudy_Sampled_Coverage_top_10000.csv")
    # fieldOfStudy_Coverage_df = fieldOfStudy_Coverage_df[fieldOfStudy_Coverage_df["do_train"] == 1].reset_index(drop=True)
    dic_results = {}
    # root_path="/media/hussein/UbuntuData/OGBN_Datasets/"
    # for i, row in fieldOfStudy_Coverage_df.iterrows():
    start_t = datetime.datetime.now()
    dataset_name=""
    # if i >= 0:
    #     for sample_key in sampledQueries:
        # dataset_name = "OBGN_MAG_Usecase_" + str(int(row["Q_idx"])) + "_" + str(str(row["topic"]).strip().replace(" ", "_").replace("/","_"))
        # dataset_name = "OBGN_MAG_"+sample_key+"Usecase_" + str(int(row["Q_idx"])) + "_" + str(str(row["topic"]).strip().replace(" ", "_").replace("/", "_"))
    # dataset_name="OBGL_DBLP_Author_Papers"
    dataset_name="OBGL_FM_DBLP_Author_Papers_filterYear_2019"
    dic_results[dataset_name] = {}
    dic_results[dataset_name]["q_idx"] = 0 #int(row["Q_idx"])
    dic_results[dataset_name]["usecase"] = dataset_name
    # dic_results[dataset_name]["sample_key"] = "starQ" #sample_key
    print("dataset=", dataset_name)
    root_path="/media/hussein/UbuntuData/OGBL_Datasets/DBLP/"
    csv_path= root_path+ dataset_name + ".tsv"
    split_by_dic={"folder_name":"time","test_valid_year":2021,"predicate":"https://dblp.org/rdf/schema#yearOfPublication"}
    if csv_path.endswith(".tsv"):
        g_tsv_df=pd.read_csv(csv_path,sep="\t")
    else:
        g_tsv_df = pd.read_csv(csv_path)
    try:
        g_tsv_df=g_tsv_df.rename(columns={"subject":"s","predicate":"p","object":"o"})
    except:
        print("g_tsv_df columns=",g_tsv_df.columns())
    ########################## filter not include predicates ############3
    g_tsv_df= g_tsv_df[~g_tsv_df["p"].isin(remove_predicates)]
    ########################delete non target papers #####################
    # lst_targets=g_tsv_df[g_tsv_df["p"]==target_rel]["s"].tolist()
    # cites_df=g_tsv_df[g_tsv_df["p"]=="http://mag.graph/cites"]
    # to_delete_papers=cites_df[~cites_df["o"].isin(lst_targets)]["o"].tolist()
    # g_tsv_df = g_tsv_df[~g_tsv_df["o"].isin(to_delete_papers)]
    # writes_df = g_tsv_df[g_tsv_df["p"] == "http://mag.graph/writes"]
    # to_delete_papers = writes_df[~writes_df["o"].isin(lst_targets)]["o"].tolist()
    # g_tsv_df=g_tsv_df[~g_tsv_df["o"].isin(to_delete_papers)]
    #####################################################################
    relations_lst=g_tsv_df["p"].unique().tolist()
    # relations_lst.remove(split_rel)
    # relations_lst.remove(target_rel)
    # if title_rel in relations_lst:
    #     relations_lst.remove(title_rel)
    ################################write relations index ########################
    relations_df=pd.DataFrame(relations_lst, columns=["rel name"])
    relations_df["rel name"]=relations_df["rel name"].apply(lambda x: str(x).split("/")[-1])
    relations_df["rel idx"]=relations_df.index
    relations_df=relations_df[["rel idx","rel name"]]
    map_folder = root_path+dataset_name+"/mapping"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    relations_df.to_csv(map_folder+"/relidx2relname.csv",index=None)
    compress_gz(map_folder+"/relidx2relname.csv")
    # # ############################### create label index ########################
    # label_idx_df= pd.DataFrame(g_tsv_df[g_tsv_df["p"] == target_rel]["o"].apply(lambda x: str(x).strip()).unique().tolist(),columns=["label name"])
    # try:
    #     label_idx_df["label name"] = label_idx_df["label name"].astype("int64")
    #     label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)
    # except:
    #     label_idx_df["label name"]=label_idx_df["label name"].astype("str")
    #     label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)
    #
    # label_idx_df["label idx"]=label_idx_df.index
    # label_idx_df=label_idx_df[["label idx","label name"]]
    # label_idx_df.to_csv(map_folder+"/labelidx2labelname.csv",index=None)
    # compress_gz(map_folder+"/labelidx2labelname.csv")
    ###########################################prepare relations mapping#################################
    relations_entites_map={}
    relations_dic={}
    entites_dic={}
    for rel in relations_lst:
        relations_dic[rel]=g_tsv_df[g_tsv_df["p"]==rel].reset_index(drop=True)
        e1=str(relations_dic[rel]["s"][0]).split("/")[3]
        e2 = str(relations_dic[rel]["o"][0]).split("/")
        if len(e2)<=1:
            e2=rel.split("/")[-1]
        else:
            e2 = e2[3]
        relations_entites_map[rel]=(e1,rel,e2)
        if e1 in entites_dic:
            entites_dic[e1]=entites_dic[e1].union(set(relations_dic[rel]["s"].apply(lambda x:str(x).replace("https://dblp.org/"+e1,"")).unique()))
        else:
            entites_dic[e1] = set(relations_dic[rel]["s"].apply(lambda x:str(x).replace("https://dblp.org/"+e1,"")).unique())

        if e2 in entites_dic:
            entites_dic[e2] = entites_dic[e2].union(set(relations_dic[rel]["o"].apply(lambda x:str(x).replace("https://dblp.org/"+e2,"")).unique()))
        else:
            entites_dic[e2] = set(relations_dic[rel]["o"].apply(lambda x:str(x).replace("https://dblp.org/"+e2,"")).unique())

    ############################ write entites index #################################
    for key in list(entites_dic.keys()) :
        try:
            entites_dic[key]=pd.DataFrame(list(entites_dic[key]), columns=['ent name']).astype('int64').sort_values(by="ent name").reset_index(drop=True)
        except:
            entites_dic[key] = pd.DataFrame(list(entites_dic[key]), columns=['ent name']).sort_values(by="ent name").reset_index(drop=True)
        entites_dic[key]=entites_dic[key].drop_duplicates()
        entites_dic[key]["ent idx"]=entites_dic[key].index
        entites_dic[key] = entites_dic[key][["ent idx","ent name"]]
        entites_dic[key+"_dic"]=pd.Series(entites_dic[key]["ent idx"].values,index=entites_dic[key]["ent name"]).to_dict()
        # print("key=",entites_dic[key+"_dic"])
        map_folder=root_path+dataset_name+"/mapping"
        try:
            os.stat(map_folder)
        except:
            os.makedirs(map_folder)
        entites_dic[key].to_csv(map_folder+"/nodeidx2name_"+key+"_id.csv",index=None)
        compress_gz(map_folder+"/nodeidx2name_"+key+"_id.csv")
    # #################### write nodes statistics ######################
    # lst_node_has_feat= [list(filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
    # lst_node_has_label=lst_node_has_feat.copy()
    # lst_num_node_dict = lst_node_has_feat.copy()
    # lst_has_feat = []
    # lst_has_label=[]
    # lst_num_node=[]
    #
    # for entity in lst_node_has_feat[0]:
    #     if str(entity)== str(label_node):
    #         lst_has_label.append("True")
    #         lst_has_feat.append("True")
    #     else:
    #         lst_has_label.append("False")
    #         lst_has_feat.append("False")
    #
    #     # lst_has_feat.append("False")
    #     lst_num_node.append( len(entites_dic[entity+"_dic"]))
    #
    # lst_node_has_feat.append(lst_has_feat)
    # lst_node_has_label.append(lst_has_label)
    # lst_num_node_dict.append(lst_num_node)
    #
    # lst_relations=[]
    #
    # for k in list(relations_entites_map.keys()):
    #     (e1,rel,e2)=relations_entites_map[k]
    #     lst_relations.append([e1,str(rel).split("/")[-1],e2])
    #
    # map_folder = root_path+dataset_name + "/raw"
    # try:
    #     os.stat(map_folder)
    # except:
    #     os.makedirs(map_folder)
    #
    # pd.DataFrame(lst_node_has_feat).to_csv(root_path+dataset_name + "/raw/nodetype-has-feat.csv", header=None, index=None)
    # compress_gz(root_path+dataset_name + "/raw/nodetype-has-feat.csv")
    #
    # pd.DataFrame(lst_node_has_label).to_csv(root_path+dataset_name + "/raw/nodetype-has-label.csv", header=None, index=None)
    # compress_gz(root_path+dataset_name + "/raw/nodetype-has-label.csv")
    #
    # pd.DataFrame(lst_num_node_dict).to_csv(root_path+dataset_name + "/raw/num-node-dict.csv", header=None, index=None)
    # compress_gz(root_path+dataset_name + "/raw/num-node-dict.csv")
    #
    # pd.DataFrame(lst_relations).to_csv(root_path+dataset_name + "/raw/triplet-type-list.csv", header=None, index=None)
    # compress_gz(root_path+dataset_name + "/raw/triplet-type-list.csv")
    # ############################### create label relation index ######################
    # label_idx_dic = pd.Series(label_idx_df["label idx"].values, index=label_idx_df["label name"]).to_dict()
    # labels_rel_df = g_tsv_df[g_tsv_df["p"] == target_rel]
    # label_type = str(labels_rel_df["s"].values[0]).split("/")
    # label_type=label_type[len(label_type)-2]
    #
    # labels_rel_df["s_idx"]=labels_rel_df["s"].apply(lambda x: str(x).split("/")[-1])
    # labels_rel_df["s_idx"] = labels_rel_df["s_idx"].astype("int64")
    # labels_rel_df["s_idx"]=labels_rel_df["s_idx"].apply(lambda x: entites_dic[label_type+"_dic"][int(x)])
    # labels_rel_df=labels_rel_df.sort_values(by=["s_idx"]).reset_index(drop=True)
    # labels_rel_df["o_idx"]=labels_rel_df["o"].apply(lambda x: str(x).split("/")[-1])
    # labels_rel_df["o_idx"]=labels_rel_df["o_idx"].apply(lambda x:label_idx_dic[int(x)])
    # out_labels_df=labels_rel_df[["o_idx"]]
    # map_folder = root_path+dataset_name + "/raw/node-label/"+label_type
    # try:
    #     os.stat(map_folder)
    # except:
    #     os.makedirs(map_folder)
    # out_labels_df.to_csv(map_folder+"/node-label.csv",header=None,index=None)
    # compress_gz(map_folder+"/node-label.csv")
    ###########################################split links (train/test/validate)###################################
    pred_node_years_df = g_tsv_df[g_tsv_df["p"] == split_by_dic["predicate"]]
    pred_node_years_df["o"]=pred_node_years_df["o"].astype("int64")
    test_valid_years_df = pred_node_years_df[pred_node_years_df["o"] == split_by_dic["test_valid_year"]]
    test_valid_papers_lst=test_valid_years_df["s"].unique().tolist()

    pred_links_df = g_tsv_df[g_tsv_df["p"] ==pred_link_predicate]
    del g_tsv_df
    gc.collect()
    pred_links_df["source_mapping_id"]=pred_links_df["s"].apply(lambda x: entites_dic["rec_dic"]["/"+str(x).split("/rec/")[-1]]) #get all papers ids
    pred_links_df["target_mapping_id"]=pred_links_df["o"].apply(lambda x: entites_dic["pid_dic"]["/"+str(x).split("/pid/")[-1]]) #get all authors ids
    distinct_target_nodes_set = set(pred_links_df["target_mapping_id"].unique().tolist())
    train_df=pred_links_df[~pred_links_df["s"].isin(test_valid_papers_lst)]
    train_df=train_df[["source_mapping_id","target_mapping_id"]]
    test_valid_df = pred_links_df[pred_links_df["s"].isin(test_valid_papers_lst)]
    test_valid_dic=test_valid_df.groupby('source_mapping_id')['target_mapping_id'].apply(list).to_dict()
    test_source_list=[]
    test_target_list=[]
    valid_source_list=[]
    valid_target_list=[]
    train_source_list=[]
    train_target_list=[]
    ##################select n-2 links for training #####################
    for key in test_valid_dic:
        valid_source_list.append(key)
        test_source_list.append(key)

        valid_target_list.append(test_valid_dic[key][0]) ## first link for valid
        if len(test_valid_dic[key])>1:
            test_target_list.append(test_valid_dic[key][1]) ## second link for test
        else:
            test_target_list.append(test_valid_dic[key][0])  ## second link for test
        if len(test_valid_dic[key]) > 2:
            for elem in test_valid_dic[key][2:]:
                train_source_list.append(key)
                train_target_list.append(elem) ## rest of links
    ##############flatting and saving train links dic#################
    train_all_source_list=train_df["source_mapping_id"].tolist()
    train_all_source_list.extend(train_source_list)
    train_all_target_list = train_df["target_mapping_id"].tolist()
    train_all_target_list.extend(train_target_list)
    traning_links_dic={"source_node":numpy.asarray(train_all_source_list,dtype=np.int64),"target_node":numpy.asarray(train_all_target_list,dtype=np.int64)}
    map_folder = root_path + dataset_name + "/split/time/"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    torch.save(traning_links_dic,map_folder+"train.pt")
    del traning_links_dic
    gc.collect()
    # compress_gz(map_folder + "train.pt")
    ##############saving valid & test links dic#################

    n_negative_samples=100
    ############build negative random samples ###############
    # num_cores = multiprocessing.cpu_count()
    # test_valid_keys = tqdm([list(test_valid_dic.keys())[i:i + 500] for i in range(0, len(list(test_valid_dic.keys())), 500)])
    # # test_valid_keys = tqdm(list(test_valid_dic.keys()))
    # Parallel(n_jobs=num_cores, backend="threading")(delayed(addNegativeSamples)(keys,n_negative_samples) for keys in test_valid_keys)
    ###########################################################
    rand_test_idxs=[random.randint(1,100000) for _ in range(len(list(test_valid_dic.keys())))]
    rand_valid_idxs = [random.randint(1, 100000) for _ in range(len(list(test_valid_dic.keys())))]
    rand_idx=0
    keys_2d =list(sliced(list(test_valid_dic.keys()), 500)) # group key into 1d lists of 500 element
    for keys_sublist in tqdm(keys_2d):
        pos_sublist=[]
        for key in keys_sublist:
            pos_sublist.extend(test_valid_dic[key]) ## collect all postive keys targets for keys sub-list
        neg_list = list(distinct_target_nodes_set - set(pos_sublist)) # get negative list for keys sub-list
        random.shuffle(neg_list) # shuffle the negative list
        for key in keys_sublist:
            rand_test_idx =rand_test_idxs[rand_idx] #get starting random index
            test_negatives_links_lst.append(neg_list[rand_test_idx-1:rand_test_idx+n_negative_samples])
            # test_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))
            # test_negatives_links_lst.append(neg_list[rand_test_idx:rand_test_idx+n_negative_samples])
            rand_valid_idx = rand_valid_idxs[rand_idx] #get starting random index
            # valid_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))
            valid_negatives_links_lst.append(neg_list[rand_valid_idx-1:rand_valid_idx+n_negative_samples])
            rand_idx+=1
    # lst_neg_sample_threads=[]
    # key_lst=list(test_valid_dic.keys())
    # for thread_idx in range (0,7):
    #     lst_neg_sample_threads.append(multiprocessing.Process(target=addNegSamples,
    #                                    args=[distinct_source_nodes_set, test_valid_dic, key_lst[thread_idx*len(key_lst)/8,len(key_lst)/8],test_negatives_links_lst, valid_negatives_links_lst, n_negative_samples]))
    #
    # for thread_idx in range (1,8):
    #     lst_neg_sample_threads[thread_idx].start()
    #     lst_neg_sample_threads[thread_idx].join()
    # print(len(valid_negatives_links_lst))

    # for key in test_valid_dic:
    #     idx+=1
    #     neg_list=list(distinct_source_nodes_set-set(test_valid_dic[key]))
    #     rand_test_idx=random.randint(0, len(neg_list)-n_negative_samples)
    #     test_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))
    #     # test_negatives_links_lst.append(neg_list[rand_test_idx:rand_test_idx+n_negative_samples])
    #     rand_valid_idx = random.randint(0, len(neg_list) - n_negative_samples)
    #     valid_negatives_links_lst.append(random.sample(neg_list, n_negative_samples))
    #     # valid_negatives_links_lst.append(neg_list[rand_valid_idx:rand_valid_idx + n_negative_samples])
    #     if idx%1000==0:
    #         print(idx)
    print("len test_negatives_links_lst=",len(test_negatives_links_lst))
    print("len valid_negatives_links_lst=", len(valid_negatives_links_lst))
    #########################################################
    test_links_dic = {"source_node": numpy.asarray(test_source_list,dtype=np.int64),
                      "target_node": numpy.asarray(test_target_list,dtype=np.int64),
                      "target_node_neg": numpy.asarray(test_negatives_links_lst,dtype=np.int64)}
    valid_links_dic = {"source_node": numpy.asarray(valid_source_list,dtype=np.int64),
                      "target_node": numpy.asarray(valid_target_list,dtype=np.int64),
                      "target_node_neg": numpy.asarray(valid_negatives_links_lst,dtype=np.int64)}
    # map_folder = root_path + dataset_name + "/split/time/test/archive/"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    # with open(map_folder+'data.pkl', 'wb') as handle:
    #     pickle.dump(test_links_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(test_links_dic, map_folder + "test.pt")
    # compress_gz(map_folder+'test.pt')

    # with open(map_folder+'data.pkl', 'wb') as handle:
    #     pickle.dump(valid_links_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # compress_gz(map_folder+'data.pkl')
    torch.save(valid_links_dic, map_folder + "valid.pt")
    # compress_gz(map_folder + 'valid.pt')

    ############################ write entites relations  #################################
    # print( list(entites_dic.keys()))
    link_rels= [pred_link_predicate]
    # for rel in relations_dic:
    for rel in link_rels:
        e1,rel,e2=relations_entites_map[rel]
        relations_dic[rel]["s_idx"]=relations_dic[rel]["s"].apply(lambda x:entites_dic[e1+"_dic"]["/"+str(x).split("/"+e1+"/")[-1]] ).astype("int64")
        relations_dic[rel]["o_idx"] = relations_dic[rel]["o"].apply(lambda x: entites_dic[e2 + "_dic"]["/"+str(x).split("/"+e2+"/")[-1]]).astype("int64")
        relations_dic[rel]=relations_dic[rel].sort_values(by="s_idx").reset_index(drop=True)
        rel_out=relations_dic[rel].drop(columns=["s","p","o"])
        # map_folder = root_path+dataset_name+"/raw/relations/"+e1+"___"+rel.split("/")[-1]+"___"+e2
        map_folder = root_path + dataset_name + "/raw"
        try:
            os.stat(map_folder)
        except:
            os.makedirs(map_folder)
        rel_out.to_csv(map_folder + "/edge.csv", index=None, header=None)
        compress_gz(map_folder + "/edge.csv")
        ########## write node num #################
        f = open(map_folder+"/num-node-list.csv", "w")
        f.write(str(len(entites_dic[e1+"_dic"])))
        f.close()
        compress_gz(map_folder+"/num-node-list.csv")
        ########## write relations num #################
        f = open(map_folder + "/num-edge-list.csv", "w")
        f.write(str(len(relations_dic[rel])))
        f.close()
        compress_gz(map_folder + "/num-edge-list.csv")
        ########## write node year  #################
        paper_years_df=relations_dic[split_rel]
        paper_years_df["s_idx"] = paper_years_df["s"].apply(lambda x: entites_dic[e1 + "_dic"]["/"+str(x).split("/rec/")[-1]]).astype("int64")
        paper_years_df=paper_years_df.sort_values(by='s_idx')
        paper_years_df=paper_years_df[["o"]]
        paper_years_df.to_csv(map_folder + "/node_year.csv", index=None, header=None)
        compress_gz(map_folder + "/node_year.csv")
        # ##################### write relations idx #######################
        # rel_idx=relations_df[relations_df["rel name"]==rel.split("/")[-1]]["rel idx"].values[0]
        # rel_out["rel_idx"]=rel_idx
        # rel_idx_df=rel_out["rel_idx"]
        # rel_idx_df.to_csv(map_folder+"/edge_reltype.csv",header=None,index=None)
        # compress_gz(map_folder+"/edge_reltype.csv")
    #####################Zip Folder ################
    shutil.make_archive(root_path+dataset_name, 'zip', root_dir=root_path,base_dir=dataset_name)
    shutil.rmtree(root_path+dataset_name)
    end_t = datetime.datetime.now()
    print("csv_to_Hetrog_time=", end_t - start_t, " sec.")
    dic_results[dataset_name]["csv_to_Hetrog_time"] = (end_t - start_t).total_seconds()

# pd.DataFrame(dic_results).transpose().to_csv("/media/hussein/UbuntuData/OGBN_Datasets/Sparql_Sampling/OGBL__Uscases_CSVtoHetrog_times" + ".csv", index=False)
    # print(entites_dic)
