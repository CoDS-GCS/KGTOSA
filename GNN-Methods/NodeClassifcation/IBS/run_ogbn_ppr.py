import logging
import resource
import time
import traceback

import os.path as osp
import numpy as np
import seml
import torch
from sacred import Experiment

from dataloaders.get_loaders_ppr import get_loaders
from data.data_preparation import check_consistence, load_data,load_data_hetero, GraphPreprocess,GraphPreprocess_hetero,GraphPreprocess_homo_hetero,load_data_hetero_homo
from models.get_model import get_model
from train.trainer_ppr import Trainer
from resource import *
ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    print("db_collection=",db_collection)
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(dataset_name,
        mode,
        batch_size,
        micro_batch,
        batch_order,
        inference,
        LBMB_val,
        small_trainingset,

        ppr_params,
        batch_params=None,
        graphmodel=None,
        hidden_channels=256,
        reg=0.,
        num_layers=3,
        heads=None,
        #epoch_min=300,
        #epoch_max=800,
        epoch_min=30,
        epoch_max=30,
        patience=100,
        lr=1e-3,
        seed=None,is_hetero=0 ):
    try:
        # dataset_name="mag"

        check_consistence(mode, batch_order)
        logging.info(f'dataset: {dataset_name}, graphmodel: {graphmodel}, mode: {mode}')
        hetero_graph, key2int,split_idx=None,None,None
        if is_hetero == 0:
            graph, (train_indices, val_indices, test_indices) = load_data(dataset_name,
                                                                          small_trainingset,
                                                                          GraphPreprocess(True, True))
        else:

            graph, (train_indices, val_indices, test_indices),hetero_graph,key2int,split_idx,local2global,subject_node_idx = load_data_hetero_homo(dataset_name,
                                                                                 small_trainingset,
                                                                                 GraphPreprocess_homo_hetero(True, True))

            # graph, (train_indices, val_indices, test_indices) = load_data_hetero(dataset_name,
            #                                                                      small_trainingset,
            #                                                                      GraphPreprocess_hetero(True, True))
            # output_node = list(graph.y_dict.keys())[0]

        logging.info("Graph loaded!\n")
        print(getrusage(RUSAGE_SELF))
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        trainer = Trainer(mode,
                          batch_params['num_batches'][0],
                          micro_batch=micro_batch,
                          batch_size=batch_size,
                          epoch_max=epoch_max,
                          epoch_min=epoch_min,
                          patience=patience)

        comment = '_'.join([dataset_name,
                            graphmodel,
                            mode])

        (train_loader,
         self_val_loader,
         self_test_loader) = get_loaders(
            graph,
            (train_indices, val_indices, test_indices),
            batch_size,
            mode,
            batch_order,
            ppr_params,
            inference,
            LBMB_val,is_hetero=0,local2global=local2global,subject_node_idx=subject_node_idx)

        stamp = ''.join(str(time.time()).split('.')) + str(seed)

        logging.info(f'model info: {comment}/model_{stamp}.pt')
        # if is_hetero==0:
        model = get_model(graphmodel,
                          graph.num_node_features,
                          graph.y.max().item() + 1,
                          hidden_channels,
                          num_layers,
                          heads,
                          device,hetero_graph,key2int)
        ######################3
        feat = torch.Tensor(graph.num_nodes, 128)
        torch.nn.init.xavier_uniform_(feat)
        x_feat = {}
        x_feat[3] = feat
        print("x_feat=", x_feat)
        #################################
        model=trainer.train(train_loader,
                      self_val_loader,
                      model=model,
                      lr=lr,
                      reg=reg,
                      comment=comment,
                      run_no=stamp,org_graph=graph,x_feat=x_feat )

        gpu_memory = torch.cuda.max_memory_allocated()
        if inference:
            # model_dir = osp.join('./saved_models', comment)
            # assert osp.isdir(model_dir)
            # model_path = osp.join(model_dir, f'model_{stamp}.pt')
            # model.load_state_dict(torch.load(model_path))
            model.eval()

            trainer.inference(self_val_loader,
                              self_test_loader,
                              model,org_graph=graph,x_feat=x_feat)


            # trainer.full_graph_inference(model, graph,train_indices,val_indices, test_indices,x_feat=x_feat,key2int=key2int,subject_node='paper')

        runtime_train_lst = []
        runtime_self_val_lst = []
        runtime_part_val_lst = []
        runtime_ppr_val_lst = []
        for curves in trainer.database['training_curves']:
            runtime_train_lst += curves['per_train_time']
            runtime_self_val_lst += curves['per_self_val_time']

        results = {
            'runtime_train_perEpoch': sum(runtime_train_lst) / len(runtime_train_lst),
            'runtime_selfval_perEpoch': sum(runtime_self_val_lst) / len(runtime_self_val_lst),
            'gpu_memory': gpu_memory,
            'max_memory': 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'curves': trainer.database['training_curves'],
            # ...
        }

        for key, item in trainer.database.items():
            if key != 'training_curves':
                results[f'{key}_record'] = item
                item = np.array(item)
                results[f'{key}_stats'] = (item.mean(), item.std(),) if len(item) else (0., 0.,)
        print(getrusage(RUSAGE_SELF))
        return results
    except:
        traceback.print_exc()
        exit()
