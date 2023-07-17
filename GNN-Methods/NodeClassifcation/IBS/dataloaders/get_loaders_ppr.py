import os
import numpy as np
import pandas as pd
from math import ceil
from typing import Dict, Tuple, Union, Optional

from torch import LongTensor

from torch_geometric.data import Data

from dataloaders.ClusterGCNLoader import ClusterGCNLoader
from dataloaders.GraphSAINTRWSampler import SaintRWTrainSampler, SaintRWValSampler
from dataloaders.IBMBBatchLoader import IBMBBatchLoader,IBMBBatchLoader_hetero
from dataloaders.IBMBNodeLoader import IBMBNodeLoader,IBMBNodeLoader_hetero
from dataloaders.IBMBRandLoader import IBMBRandLoader
from dataloaders.IBMBRandfixLoader import IBMBRandfixLoader
from dataloaders.ShaDowLoader import ShaDowLoader
from dataloaders.LADIESSampler import LADIESSampler
from dataloaders.NeighborSamplingLoader import NeighborSamplingLoader

Loader = Union[
    IBMBNodeLoader,
]
EDGE_INDEX_TYPE = 'adj'


def num_out_nodes_per_batch_normalization(num_out_nodes: int,
                                          num_out_per_batch: int):
    num_batches = ceil(num_out_nodes / num_out_per_batch)
    return ceil(num_out_nodes / num_batches)


def get_loaders(graph: Data,
                splits: Tuple[LongTensor, LongTensor, LongTensor],
                batch_size: int,
                mode: str,
                batch_order: str,
                ppr_params: Optional[Dict],
                inference: bool = True,
                ibmb_val: bool = True,is_hetero=0,local2global=None,subject_node_idx=None) -> Tuple[
    Optional[Loader]
]:
    train_indices, val_indices, test_indices = splits

    train_loader = None
    self_val_loader = None
    self_test_loader = None
    if mode == 'ppr':
        if is_hetero==0:
            train_loader = IBMBNodeLoader(graph,
                                      batch_order,
                                      train_indices if local2global ==None else local2global[subject_node_idx][train_indices],
                                      EDGE_INDEX_TYPE,
                                      ppr_params['neighbor_topk'],
                                      num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                          len(train_indices), ppr_params['primes_per_batch']),
                                      num_auxiliary_nodes_per_batch=None,
                                      alpha=ppr_params['alpha'],
                                      eps=ppr_params['eps'],
                                      batch_size=batch_size,
                                      shuffle=False)    # must be false, instead we define our own order!
            self_val_loader = IBMBNodeLoader(graph,
                                             batch_order,
                                             val_indices if local2global ==None else local2global[subject_node_idx][val_indices],
                                             EDGE_INDEX_TYPE,
                                             ppr_params['neighbor_topk'],
                                             num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                                 len(val_indices), ppr_params['primes_per_batch'] * 2),
                                             num_auxiliary_nodes_per_batch=None,
                                             alpha=ppr_params['alpha'],
                                             eps=ppr_params['eps'],
                                             batch_size=batch_size,
                                             shuffle=False)
            if inference:
                self_test_loader = IBMBNodeLoader(graph,
                                                  batch_order,
                                                  test_indices if local2global ==None else local2global[subject_node_idx][test_indices],
                                                  EDGE_INDEX_TYPE,
                                                  ppr_params['neighbor_topk'],
                                                  num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                                      len(test_indices), len(test_indices) ),
                                                  # len(test_indices), ppr_params['primes_per_batch'] * 2),
                                                  num_auxiliary_nodes_per_batch=None,
                                                  alpha=ppr_params['alpha'],
                                                  eps=ppr_params['eps'],
                                                  batch_size=batch_size,
                                                  shuffle=False)
        else:
            train_loader = IBMBNodeLoader_hetero(graph,
                                          batch_order,
                                          train_indices,
                                          EDGE_INDEX_TYPE,
                                          ppr_params['neighbor_topk'],
                                          num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                              len(train_indices), ppr_params['primes_per_batch']),
                                          num_auxiliary_nodes_per_batch=None,
                                          alpha=ppr_params['alpha'],
                                          eps=ppr_params['eps'],
                                          batch_size=batch_size,
                                          shuffle=False)  # must be false, instead we define our own order!
            self_val_loader = IBMBNodeLoader_hetero(graph,
                                             batch_order,
                                             val_indices,
                                             EDGE_INDEX_TYPE,
                                             ppr_params['neighbor_topk'],
                                             num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                                 len(val_indices), ppr_params['primes_per_batch'] * 2),
                                             num_auxiliary_nodes_per_batch=None,
                                             alpha=ppr_params['alpha'],
                                             eps=ppr_params['eps'],
                                             batch_size=batch_size,
                                             shuffle=False)
            if inference:
                self_test_loader = IBMBNodeLoader_hetero(graph,
                                                  batch_order,
                                                  test_indices,
                                                  EDGE_INDEX_TYPE,
                                                  ppr_params['neighbor_topk'],
                                                  num_output_nodes_per_batch=num_out_nodes_per_batch_normalization(
                                                      len(test_indices), ppr_params['primes_per_batch'] * 2),
                                                  num_auxiliary_nodes_per_batch=None,
                                                  alpha=ppr_params['alpha'],
                                                  eps=ppr_params['eps'],
                                                  batch_size=batch_size,
                                                  shuffle=False)
    return (train_loader,
            self_val_loader,
            self_test_loader)
