import numpy as np
from scipy.stats import dirichlet
from scipy import stats
from dataclasses import dataclass
import itertools
import random
from omegaconf import OmegaConf
from pathlib import Path
import torch.nn.functional as F

from dataclasses import dataclass
import itertools
import logging
import random
import math
import pickle
import time
import torch
import sys

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple

from dataset import DataArgs, Dataset, iterate_batches
from model import ModelArgs, SelfAttention
import argparse
from misc import get_filenames, angle_estimate, rank_k_approximation
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
from matplotlib.font_manager import FontProperties

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 6),
         'axes.labelsize': 'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

font = FontProperties()
font.set_family('serif')
font.set_size('12')

def get_inner_hamming(filenames, n_concept):

    values = np.zeros((len(filenames), n_concept+1))
    
    eval_list = []
    for exp_i, filename in enumerate(filenames):
        hamming_cos_dict = {}

        # create_one_hot_encoding(args)
        checkpoint = torch.load(filename)

        ds = checkpoint['ds']

        cfg = checkpoint['config']

        cfg.model_args.vocab_size = ds.vocab_size

        model = SelfAttention(cfg.model_args,
                vocab_size=cfg.model_args.vocab_size)

        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        cfg.max_iters = 1

        effective_vocab_size = ds.vocab_size - 1
        heat_map = np.zeros((effective_vocab_size, effective_vocab_size))
        for i, (x, y) in enumerate(iterate_batches(ds, batch_size=1024,
                                                            num_workers=cfg.num_data_workers)):
            if cfg.max_iters is not None and i >= cfg.max_iters:
                break

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

            pred = model(x)

            eval_acc = (pred[:,-1,:].argmax(-1) == y[:,-1]).float().mean().item()

            eval_list.append(eval_acc)

        unembedding = model.output.weight.detach().numpy()

        # singular_values = np.linalg.svd(unembedding, compute_uv=False)

        # Plot the spectrum
        # plt.figure()
        # plt.plot(singular_values, 'bo')
        # plt.ylabel('Singular Value')
        # plt.title('Spectrum of the Unembeddings')
        # plt.grid(True)
        # plt.savefig("spectrum_n" + str(ds.n_concept) + "_d" + str(cfg.model_args.dim) + "_e" + str(exp_i) + ".png")

        all_target_vectors = np.array(list(itertools.product([0, 1], repeat=ds.n_concept)))

        diff_list = []
        for base_ind in range(2**ds.n_concept):
            diff_list.append(unembedding[base_ind,:])
        cos_matrix_unembed = cosine_similarity(np.vstack(diff_list))

        inner_unembed = np.vstack(diff_list) @ np.vstack(diff_list).T

        
        for target in all_target_vectors:
            all_vectors = np.array(list(itertools.product([0, 1], repeat=ds.n_concept)))
            
            for vector in all_vectors:
                if ds.tokenize(target) >= ds.tokenize(vector):
                    # if vector[0] == target[0]:
                    hamming_distance = np.count_nonzero(target!=vector)
                    cos_list = hamming_cos_dict.get(hamming_distance, [])
                    cos_list.append(inner_unembed[int(ds.tokenize(target)), int(ds.tokenize(vector))])
                    # cos_list.append(np.linalg.norm([unembedding[int(ds.tokenize(target))] - unembedding[int(ds.tokenize(vector))]]))
                    hamming_cos_dict[hamming_distance]=cos_list

        # values_array = np.zeros(len(hamming_cos_dict))
        # ste_array = np.zeros(len(hamming_cos_dict))
        for key, value in hamming_cos_dict.items():
            
            # values_array[key] = np.average(value)
            # ste_array[key] = stats.sem(value)
            # ste_array[key] = np.std(value)
            values[exp_i, key] = np.average(value)


    # return values_array, ste_array
    return np.average(values, axis=0), stats.sem(values, axis=0)

# def get_inner_hamming(filenames):
#     hamming_cos_dict = {}
#     eval_list = []
#     for exp_i, filename in enumerate(filenames):

#         # create_one_hot_encoding(args)
#         checkpoint = torch.load(filename)

#         ds = checkpoint['ds']

#         cfg = checkpoint['config']

#         cfg.model_args.vocab_size = ds.vocab_size

#         model = SelfAttention(cfg.model_args,
#                 vocab_size=cfg.model_args.vocab_size)

#         model.load_state_dict(checkpoint['model_state_dict'])

#         model.eval()
#         cfg.max_iters = 1

#         effective_vocab_size = ds.vocab_size - 1
#         heat_map = np.zeros((effective_vocab_size, effective_vocab_size))
#         for i, (x, y) in enumerate(iterate_batches(ds, batch_size=1024,
#                                                             num_workers=cfg.num_data_workers)):
#             if cfg.max_iters is not None and i >= cfg.max_iters:
#                 break

#             x = torch.from_numpy(x)
#             y = torch.from_numpy(y)

#             pred = model(x)

#             eval_acc = (pred[:,-1,:].argmax(-1) == y[:,-1]).float().mean().item()

#             eval_list.append(eval_acc)

#         unembedding = model.output.weight.detach().numpy()

#         # singular_values = np.linalg.svd(unembedding, compute_uv=False)

#         # Plot the spectrum
#         # plt.figure()
#         # plt.plot(singular_values, 'bo')
#         # plt.ylabel('Singular Value')
#         # plt.title('Spectrum of the Unembeddings')
#         # plt.grid(True)
#         # plt.savefig("spectrum_n" + str(ds.n_concept) + "_d" + str(cfg.model_args.dim) + "_e" + str(exp_i) + ".png")

#         all_target_vectors = np.array(list(itertools.product([0, 1], repeat=ds.n_concept)))

#         diff_list = []
#         for base_ind in range(2**ds.n_concept):
#             diff_list.append(unembedding[base_ind,:])
#         cos_matrix_unembed = cosine_similarity(np.vstack(diff_list))

#         inner_unembed = np.vstack(diff_list) @ np.vstack(diff_list).T

        
#         for target in all_target_vectors:
#             all_vectors = np.array(list(itertools.product([0, 1], repeat=ds.n_concept)))
            
#             ds.tokenize(target)
#             for vector in all_vectors:
#                 if ds.tokenize(target) >= ds.tokenize(vector):
#                     # if vector[0] == target[0]:
#                     hamming_distance = np.count_nonzero(target!=vector)
#                     cos_list = hamming_cos_dict.get(hamming_distance, [])
#                     cos_list.append(inner_unembed[int(ds.tokenize(target)), int(ds.tokenize(vector))])
#                     # cos_list.append(np.linalg.norm([unembedding[int(ds.tokenize(target))] - unembedding[int(ds.tokenize(vector))]]))
#                     hamming_cos_dict[hamming_distance]=cos_list

#     values_array = np.zeros(len(hamming_cos_dict))
#     ste_array = np.zeros(len(hamming_cos_dict))
#     for key, value in hamming_cos_dict.items():
        
#         values_array[key] = np.average(value)
#         ste_array[key] = stats.sem(value)
#         # ste_array[key] = np.std(value)

#     return values_array, ste_array


if __name__ == '__main__':

    @dataclass
    class OptimArgs:
        learning_rate: float = 0.01
        weight_decay: float = 0.01
        momentum: float = 0.9  # for SGD
        batch_size: int = 256
        use_sgd: bool = False  # otherwise use AdamW

    @dataclass
    class TrainerArgs:
        optim_args: OptimArgs
        data_args: DataArgs
        model_args: ModelArgs
        max_iters: Optional[int] = None
        eval_delta: int = 5
        num_data_workers: int = 5
        save_dir: Optional[str] = None
        root_dir: str = '/net/scratch/yiboj/mem-llm'
        config_file: Optional[str] = None
        exp_id: int=1
        eval: bool = False
        eval_path: Optional[str] = None
        device: str="cpu"

    args = TrainerArgs(
            optim_args=OptimArgs(),
            data_args=DataArgs(),
            model_args=ModelArgs()
            )

    cfg_before_merge = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())
    if cfg_before_merge.config_file is not None:
        cfg = OmegaConf.merge(cfg_before_merge, OmegaConf.load(cfg_before_merge.config_file), OmegaConf.from_cli())
    else:
        cfg = cfg_before_merge

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.save_dir is not None:
        outdir = Path(cfg.root_dir) / Path(cfg.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)

    dims = [32, 64, 96]
    # dims = [96]
    for n_concept in [5, 6, 7, 8]:
        plt.figure()
        for dim in dims:
            # cfg.eval_path = "./saved_results/low-len-128/no_wv_low_dim" + str(n_concept)
            cfg.eval_path = "./saved_results/low-len-128-d"+ str(dim) +"/no_wv_low_dim" + str(n_concept)
            filenames = get_filenames(cfg.eval_path)

            values_array, ste_array = get_inner_hamming(filenames, n_concept)

            plt.ylabel("Inner Products", fontsize=20)
            plt.xlabel("Hamming Distance", fontsize=20)

            sns.lineplot(x=np.arange(0, n_concept+1), y=values_array, label="Dim = " + str(dim))
            plt.fill_between(np.arange(0, n_concept+1), values_array - ste_array, values_array + ste_array, alpha=0.2)

        plt.legend()    
        plt.savefig("no_wv_hamming_inner_n" + str(n_concept) + ".png")