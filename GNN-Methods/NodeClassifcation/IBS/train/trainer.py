import logging
import os
import time
from collections import defaultdict
from math import ceil

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_sparse import SparseTensor

from dataloaders.BaseLoader import BaseLoader
from .prefetch_generators import BackgroundGenerator
from .train_utils import run_batch,run_rgcn_batch
from data.data_utils import MyGraph
from tqdm import tqdm
from models.RGCN import rgcn_test
# from torch.utils.tensorboard import SummaryWriter

device_type='cpu'
class Trainer:
    def __init__(self,
                 mode: str,
                 num_batches: int,
                 micro_batch: int = 1,
                 batch_size: int = 1,
                 epoch_max: int = 800,
                 epoch_min: int = 300,
                 patience: int = 100,
                 device: str = device_type):

        super().__init__()

        self.mode = mode
        self.device = device
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.micro_batch = micro_batch
        self.epoch_max = epoch_max
        self.epoch_min = epoch_min
        self.patience = patience

        self.database = defaultdict(list)

    def get_loss_scaling(self, len_loader: int):
        if type(len_loader)==int:
            micro_batch = int(min(self.micro_batch, len_loader))
            loss_scaling_lst = [micro_batch] * (len_loader // micro_batch) + [len_loader % micro_batch]
        else:
            micro_batch = int(min(self.micro_batch, len_loader()))
            loss_scaling_lst = [micro_batch] * (len_loader() // micro_batch) + [len_loader() % micro_batch]
        return loss_scaling_lst, micro_batch

    def train(self,
              train_loader,
              self_val_loader,
              ppr_val_loader,
              batch_val_loader,
              model,
              lr,
              reg,
              comment='',
              run_no='',org_graph=None,x_feat=None):

        #         writer = SummaryWriter('./runs')
        patience_count = 0
        best_accs = {'train': 0., 'self': 0., 'part': 0., 'ppr': 0.}
        best_val_acc = -1.

        if not os.path.isdir('./saved_models'):
            os.mkdir('./saved_models')
        model_dir = os.path.join('./saved_models', comment)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_path = os.path.join(model_dir, f'model_{run_no}.pt')

        # start training
        training_curve = defaultdict(list)

        # opt = torch.optim.Adam(model.p_list, lr=lr, weight_decay=reg)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.33, patience=30,
                                                               cooldown=10, min_lr=1e-4)
       ###################
        max_subgraphs=50
        ###############
        next_loader = BackgroundGenerator(train_loader)
        for epoch in range(self.epoch_max):
            logging.info(f"Epoch {epoch}")
            data_dic = {'self': {'loss': 0., 'acc': 0., 'num': 0},
                        'part': {'loss': 0., 'acc': 0., 'num': 0},
                        'train': {'loss': 0., 'acc': 0., 'num': 0},
                        'ppr': {'loss': 0., 'acc': 0., 'num': 0}, }

            update_count = 0

            # train
            model.train()
            loss_scaling_lst, cur_micro_batch = self.get_loss_scaling(train_loader.loader_len)
            loader, next_loader = next_loader, None

            start_time = time.time()
            # print ("len(loader)=",loader.len())
            pbar = tqdm(total=max_subgraphs)
            pbar.set_description(f"Train Loader Progress")
            subg_idx=0
            while True:
                data = loader.next()
                if subg_idx == max_subgraphs:
                    if batch_val_loader is not None:
                        next_loader = BackgroundGenerator(batch_val_loader)
                    elif ppr_val_loader is not None:
                        next_loader = BackgroundGenerator(ppr_val_loader)
                    else:
                        next_loader = BackgroundGenerator(self_val_loader)
                    break
                if data is None:
                    loader = None
                    loss = None
                    corrects = None
                    break
                else:
                    if data[1]:  # stop signal
                        if batch_val_loader is not None:
                            next_loader = BackgroundGenerator(batch_val_loader)
                        elif ppr_val_loader is not None:
                            next_loader = BackgroundGenerator(ppr_val_loader)
                        else:
                            next_loader = BackgroundGenerator(self_val_loader)
                if str(type(model)).split('.')[1]=='RGCN':
                    opt.zero_grad()
                    loss, corrects, num_nodes,pred,y,acc = run_rgcn_batch(model, data[0], loss_scaling_lst[0],org_graph,x=x_feat)
                    # print('Train Acc=', acc)
                    opt.step()
                    pbar.update(1)
                else:
                    loss, corrects, num_nodes, _, _ = run_batch(model, data[0], loss_scaling_lst[0])
                    pbar.update(1)
                data_dic['train']['loss'] += loss
                data_dic['train']['acc'] += corrects
                data_dic['train']['num'] += num_nodes
                update_count += 1
                # print("num_nodes=",num_nodes,"corrects=",corrects)

                if update_count >= cur_micro_batch:
                    loss_scaling_lst.pop(0)
                    opt.step()
                    opt.zero_grad()
                    update_count = 0
                subg_idx+=1
            pbar.close()

            # remainder
            if update_count:
                opt.step()
                opt.zero_grad()

            train_time = time.time() - start_time

            logging.info('After train loader -- '
                         f'Allocated: {torch.cuda.memory_allocated()}, '
                         f'Max allocated: {torch.cuda.max_memory_allocated()}, '
                         f'Reserved: {torch.cuda.memory_reserved()}')

            model.eval()

            # part val first, for fairness of all methods
            start_time = time.time()
            if batch_val_loader is not None:
                loader, next_loader = next_loader, None
                subg_idx = 0
                pbar = tqdm(total=max_subgraphs)
                pbar.set_description(f"Valid Loader Progress")
                while True:
                    data = loader.next()
                    if subg_idx == max_subgraphs:
                        if ppr_val_loader is not None:
                            next_loader = BackgroundGenerator(ppr_val_loader)
                        else:
                            next_loader = BackgroundGenerator(self_val_loader)
                        break
                    if data is None:
                        loader = None
                        loss = None
                        corrects = None
                        break
                    else:
                        if data[1]:  # stop signal
                            if ppr_val_loader is not None:
                                next_loader = BackgroundGenerator(ppr_val_loader)
                            else:
                                next_loader = BackgroundGenerator(self_val_loader)

                    with torch.no_grad():
                        if str(type(model)).split('.')[1] == 'RGCN':
                            loss, corrects, num_nodes, pred,y,acc = run_rgcn_batch(model, data[0], loss_scaling_lst[0],org_graph,x=x_feat)
                            # print('Valid Acc=',acc)
                            pbar.update(1)
                        else:
                            loss, corrects, num_nodes, _, _ = run_batch(model, data[0])
                            pbar.update(1)
                        data_dic['part']['loss'] += loss
                        data_dic['part']['acc'] += corrects
                        data_dic['part']['num'] += num_nodes

                    subg_idx += 1
                pbar.close()

            part_val_time = time.time() - start_time

            # ppr val
            start_time = time.time()
            if ppr_val_loader is not None:
                loader, next_loader = next_loader, None
                subg_idx = 0
                pbar = tqdm(total=max_subgraphs)
                pbar.set_description(f"ppr val Loader Progress")
                while True:
                    data = loader.next()
                    if subg_idx == max_subgraphs:
                        next_loader = BackgroundGenerator(self_val_loader)
                        break
                    if data is None:
                        loader = None
                        loss = None
                        corrects = None
                        break
                    else:
                        if data[1]:  # stop signal
                            next_loader = BackgroundGenerator(self_val_loader)

                    with torch.no_grad():
                        if str(type(model)).split('.')[1] == 'RGCN':
                            loss, corrects, num_nodes, pred,y,acc = run_rgcn_batch(model, data[0], loss_scaling_lst[0],org_graph,x=x_feat)
                            # print('ppr Acc=', acc)
                            pbar.update(1)
                        else:
                            loss, corrects, num_nodes, _, _ = run_batch(model, data[0])
                            pbar.update(1)
                        data_dic['ppr']['loss'] += loss
                        data_dic['ppr']['acc'] += corrects
                        data_dic['ppr']['num'] += num_nodes
                    subg_idx += 1
                pbar.close()

            ppr_val_time = time.time() - start_time

            # original val
            loader, next_loader = next_loader, None
            start_time = time.time()

            subg_idx = 0
            pbar = tqdm(total=max_subgraphs)
            pbar.set_description(f"original val Loader Progress")
            while True:
                data = loader.next()
                if subg_idx==max_subgraphs:
                    if epoch < self.epoch_max - 1:
                        next_loader = BackgroundGenerator(train_loader)
                    else:
                        next_loader = None
                    break
                if data is None:
                    loader = None
                    loss = None
                    corrects = None
                    break
                else:
                    if data[1]:  # stop signal
                        if epoch < self.epoch_max - 1:
                            next_loader = BackgroundGenerator(train_loader)
                        else:
                            next_loader = None

                with torch.no_grad():
                    if str(type(model)).split('.')[1] == 'RGCN':
                        loss, corrects, num_nodes, pred,y,acc = run_rgcn_batch(model, data[0], loss_scaling_lst[0], org_graph,x=x_feat)
                        # print('original val  Acc=', acc)
                        pbar.update(1)
                    else:
                        loss, corrects, num_nodes, _, _ = run_batch(model, data[0])
                        pbar.update(1)
                    data_dic['self']['loss'] += loss
                    data_dic['self']['acc'] += corrects
                    data_dic['self']['num'] += num_nodes
                subg_idx+=1
            pbar.close()

            self_val_time = time.time() - start_time

            # update training info
            for cat in ['train', 'self', 'part', 'ppr']:
                if data_dic[cat]['num'] > 0:
                    data_dic[cat]['loss'] = (data_dic[cat]['loss'] / data_dic[cat]['num']).item()
                    data_dic[cat]['acc'] = (data_dic[cat]['acc'] / data_dic[cat]['num']).item()
                best_accs[cat] = max(best_accs[cat], data_dic[cat]['acc'])

            # lr scheduler
            criterion_val_loss = data_dic['part']['loss'] if data_dic['part']['loss'] != 0 else data_dic['self']['loss']
            if scheduler is not None:
                scheduler.step(criterion_val_loss)

            # early stop
            criterion_val_acc = data_dic['part']['acc'] if data_dic['part']['acc'] != 0 else data_dic['self']['acc']
            if criterion_val_acc > best_val_acc:
                best_val_acc = criterion_val_acc
                patience_count = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_count += 1
                if epoch > self.epoch_min and patience_count > self.patience:
                    scheduler = None
                    opt = None
                    assert loader is None

                    if next_loader is not None:
                        next_loader.stop_signal = True
                        while next_loader.is_alive():
                            batch = next_loader.next()
                        next_loader = None
                    torch.cuda.empty_cache()
                    break

            logging.info(f"train_acc: {data_dic['train']['acc']:.5f}, "
                         f"self_val_acc: {data_dic['self']['acc']:.5f}, "
                         f"part_val_acc: {data_dic['part']['acc']:.5f}, "
                         f"ppr_val_acc: {data_dic['ppr']['acc']:.5f}, "
                         f"lr: {opt.param_groups[0]['lr']}, "
                         f"patience: {patience_count} / {self.patience}\n")

            # maintain curves
            training_curve['per_train_time'].append(train_time)
            training_curve['per_self_val_time'].append(self_val_time)
            training_curve['per_part_val_time'].append(part_val_time)
            training_curve['per_ppr_val_time'].append(ppr_val_time)
            training_curve['train_loss'].append(data_dic['train']['loss'])
            training_curve['train_acc'].append(data_dic['train']['acc'])
            training_curve['self_val_loss'].append(data_dic['self']['loss'])
            training_curve['self_val_acc'].append(data_dic['self']['acc'])
            training_curve['ppr_val_loss'].append(data_dic['ppr']['loss'])
            training_curve['ppr_val_acc'].append(data_dic['ppr']['acc'])
            training_curve['part_val_loss'].append(data_dic['part']['loss'])
            training_curve['part_val_acc'].append(data_dic['part']['acc'])
            training_curve['lr'].append(opt.param_groups[0]['lr'])

        #             writer.add_scalar('train_loss', data_dic['train']['loss'], epoch)
        #             writer.add_scalar('train_acc', data_dic['train']['acc'], epoch)
        #             writer.add_scalar('self_val_loss', data_dic['self']['loss'], epoch)
        #             writer.add_scalar('self_val_acc', data_dic['self']['acc'], epoch)

        # ending
        self.database['best_train_accs'].append(best_accs['train'])
        self.database['training_curves'].append(training_curve)

        logging.info(f"best train_acc: {best_accs['train']}, "
                     f"best self val_acc: {best_accs['self']}, "
                     f"best part val_acc: {best_accs['part']}"
                     f"best ppr val_acc: {best_accs['ppr']}")

        torch.cuda.empty_cache()
        # assert next_loader is None and loader is None

    #         writer.flush()
        return model

    def train_single_batch(self,
                           dataset,
                           model,
                           lr,
                           reg,
                           val_per_epoch=5,
                           comment='',
                           run_no=''):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  self_val_loader,
                  ppr_val_loader,
                  batch_val_loader,
                  self_test_loader,
                  ppr_test_loader,
                  batch_test_loader,
                  model,
                  record_numbatch=False,org_graph=None,x_feat=None):

        cat_dict = {('self', 'val',): [self.database['self_val_accs'], self.database['self_val_f1s']],
                    ('part', 'val',): [self.database['part_val_accs'], self.database['part_val_f1s']],
                    ('ppr', 'val',): [self.database['ppr_val_accs'], self.database['ppr_val_f1s']],
                    ('self', 'test',): [self.database['self_test_accs'], self.database['self_test_f1s']],
                    ('part', 'test',): [self.database['part_test_accs'], self.database['part_test_f1s']],
                    ('ppr', 'test',): [self.database['ppr_test_accs'], self.database['ppr_test_f1s']], }

        loader_dict = {'val': {'self': self_val_loader, 'part': batch_val_loader, 'ppr': ppr_val_loader},
                       'test': {'self': self_test_loader, 'part': batch_test_loader, 'ppr': ppr_test_loader}}

        time_dict = {'self': self.database['self_inference_time'],
                     'part': self.database['part_inference_time'],
                     'ppr': self.database['ppr_inference_time']}

        print("start inference")
        for cat in ['val', 'test']:
            for sample in ['self', 'part', 'ppr']:
                acc, f1 = 0., 0.
                num_batch = 0
                if device_type=='cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                if loader_dict[cat][sample] is not None:
                    loader = BackgroundGenerator(loader_dict[cat][sample])

                    pred_labels = []
                    true_labels = []


                    max_subgraphs=100
                    pbar = tqdm(total=max_subgraphs)
                    pbar.set_description(f"inference Progress "+cat+"-"+sample)
                    while True:
                        data = loader.next()
                        if data is None:
                            del loader
                            break
                        if str(type(model)).split('.')[1] == 'RGCN':
                            loss, corrects, num_nodes, pred_label_batch, true_label_batch,acc = run_rgcn_batch(model, data[0],org_graph,x=x_feat)
                            if sample=='self':
                                print("graph=",data[0])
                            pbar.update(1)
                            # print('inference Acc=', acc)
                        else:
                            _, _, _, pred_label_batch, true_label_batch = run_batch(model, data[0])
                        pred_labels.append(pred_label_batch.detach())
                        true_labels.append(true_label_batch.detach())
                        num_batch += 1
                    pbar.close()

                    pred_labels = torch.cat(pred_labels, dim=0).cpu().numpy()
                    true_labels = torch.cat(true_labels, dim=0).cpu().numpy()

                    acc = (pred_labels == true_labels).sum() / len(true_labels)
                    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

                cat_dict[(sample, cat,)][0].append(acc)
                cat_dict[(sample, cat,)][1].append(f1)

                if record_numbatch:
                    self.database[f'numbatch_{sample}_{cat}'].append(num_batch)

                logging.info("{}_{}_acc: {:.3f}, {}_{}_f1: {:.3f}, ".format(sample, cat, acc, sample, cat, f1))
                if cat == 'test':
                    if device_type=='cuda':
                        torch.cuda.synchronize()
                    time_dict[sample].append(time.time() - start_time)

    @torch.no_grad()
    def full_graph_inference(self,
                             model,
                             graph,
                             train_nodes,
                             val_nodes,
                             test_nodes,x_feat=None,key2int=None,subject_node='Paper' ):

        if isinstance(val_nodes, torch.Tensor):
            val_nodes = val_nodes.numpy()
        if isinstance(test_nodes, torch.Tensor):
            test_nodes = test_nodes.numpy()

        start_time = time.time()

        mask = np.union1d(val_nodes, test_nodes)
        val_mask = np.in1d(mask, val_nodes)
        test_mask = np.in1d(mask, test_nodes)
        assert np.all(np.invert(val_mask) == test_mask)


        adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        adj = BaseLoader.normalize_adjmat(adj, normalization='sym')

        if str(type(model)).split('.')[1] == 'RGCN':
            train_acc, valid_acc, test_acc=rgcn_test(model, x_feat, graph, key2int, train_nodes,val_nodes,test_nodes,subject_node)

            # nodes = val_nodes if cat == 'val' else test_nodes
            # _mask = val_mask if cat == 'val' else test_mask
            # f1 = f1_score(true, pred, average='macro', zero_division=0)
            self.database[f'full_RGCN_train_acc'].append(train_acc)
            self.database[f'full_RGCN_valid_acc'].append(valid_acc)
            self.database[f'full_RGCN_test_acc'].append(test_acc)
        else:
            outputs = model.chunked_pass(MyGraph(x=graph.x, adj=adj, idx=torch.from_numpy(mask)),
                                         self.num_batches // self.batch_size).detach().numpy()  # an estimate of #chunks
            for cat in ['val', 'test']:
                nodes = val_nodes if cat == 'val' else test_nodes
                _mask = val_mask if cat == 'val' else test_mask
                pred = np.argmax(outputs[_mask], axis=1)
                true = graph.y.numpy()[nodes]

                acc = (pred == true).sum() / len(true)
                f1 = f1_score(true, pred, average='macro', zero_division=0)

                self.database[f'full_{cat}_accs'].append(acc)
                self.database[f'full_{cat}_f1s'].append(f1)

                logging.info("full_{}_acc: {:.3f}, full_{}_f1: {:.3f}, ".format(cat, acc, cat, f1))

        self.database['full_inference_time'].append(time.time() - start_time)

    @torch.no_grad()
    def full_graph_inference_hetero(self,
                             model,
                             graph,
                             val_nodes,
                             test_nodes, ):

        if isinstance(val_nodes, torch.Tensor):
            val_nodes = val_nodes.numpy()
        if isinstance(test_nodes, torch.Tensor):
            test_nodes = test_nodes.numpy()

        start_time = time.time()

        mask = np.union1d(val_nodes, test_nodes)
        val_mask = np.in1d(mask, val_nodes)
        test_mask = np.in1d(mask, test_nodes)
        assert np.all(np.invert(val_mask) == test_mask)

        output_node=list(graph.y_dict.keys())[0]
        for (s, p, o) in graph.edge_index_dict.keys():
            if (s == output_node):
                adj = SparseTensor.from_edge_index(graph.edge_index_dict[(s, p, o)], sparse_sizes=(
                graph.edge_index_dict[(s, p, o)].max() + 1, graph.edge_index_dict[(s, p, o)].max() + 1))
                adj = BaseLoader.normalize_adjmat(adj, normalization='sym')
                break ## ToDos get outputs based on graph not single adj

        outputs = model.chunked_pass(MyGraph(x=graph.x_dict[output_node], adj=adj, idx=torch.from_numpy(mask)),
                                     self.num_batches // self.batch_size).detach().numpy()  # an estimate of #chunks

        for cat in ['val', 'test']:
            nodes = val_nodes if cat == 'val' else test_nodes
            _mask = val_mask if cat == 'val' else test_mask
            pred = np.argmax(outputs[_mask], axis=1)
            true = graph.y_dict[output_node].numpy()[nodes]

            acc = (pred == true).sum() / len(true)
            f1 = f1_score(true, pred, average='macro', zero_division=0)

            self.database[f'full_{cat}_accs'].append(acc)
            self.database[f'full_{cat}_f1s'].append(f1)

            logging.info("full_{}_acc: {:.3f}, full_{}_f1: {:.3f}, ".format(cat, acc, cat, f1))

        self.database['full_inference_time'].append(time.time() - start_time)
