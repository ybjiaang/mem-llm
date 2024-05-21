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

def get_accuracy_stderr(filenames):

    eval_list = []
    for exp_i, filename in enumerate(filenames):

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

        for i, (x, y) in enumerate(iterate_batches(ds, batch_size=1024,
                                                            num_workers=cfg.num_data_workers)):
            if cfg.max_iters is not None and i >= cfg.max_iters:
                break

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

            pred = model(x)

            eval_acc = (pred[:,-1,:].argmax(-1) == y[:,-1]).float().mean().item()

            eval_list.append(eval_acc)

    # return values_array, ste_array
    return np.average(eval_list), stats.sem(eval_list)


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

    dims = [16, 32, 64]
    lengths = [8, 16, 32, 64, 128]
    for n_concept in [5, 6, 7, 8]:
        plt.figure()
        for dim in dims:
            len_acc = []
            len_ste = []
            for length in lengths:
                lr_acc = []
                lr_ste = []
                for lr in [0.01, 0.001]:
                    cfg.eval_path = "./saved_results/length/length_n" + str(n_concept) + "_d" + str(dim) + "_lr" + str(lr) + "_l" + str(length)
                    filenames = get_filenames(cfg.eval_path)

                    acc, stderr = get_accuracy_stderr(filenames)
                    lr_acc.append(acc)
                    lr_ste.append(stderr)

                if lr_acc[0] > lr_acc[1]:
                    acc = lr_acc[0]
                    ste = lr_ste[0]
                else:
                    acc = lr_acc[1]
                    ste = lr_ste[1]

                len_acc.append(acc)
                len_ste.append(ste)
        
            # Plot with error band
            # sns.lineplot(x=np.array(lengths), y=np.array(len_acc), label="Dim = " + str(dim))
            # plt.fill_between(np.array(lengths), np.array(len_acc) - np.array(len_ste), np.array(len_acc) + np.array(len_ste), alpha=0.2)
            sns.lineplot(x=np.array(lengths), y=np.array(len_acc), label="Dim = " + str(dim))
            plt.fill_between(np.array(lengths), np.array(len_acc) - np.array(len_ste), np.array(len_acc) + np.array(len_ste), alpha=0.2)

        plt.xlabel('Context Lengths', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.grid(True)
        plt.xscale('log')
        plt.xticks([8, 16, 32, 64, 128], ['8', '16', '32', '64', '128'])
        plt.minorticks_off()
        plt.legend()
        plt.tight_layout()
        plt.savefig("length_n"+str(n_concept)+".png")

            

        #     plt.ylabel("Inner Products", fontsize=20)
        #     plt.xlabel("Hamming Distance", fontsize=20)

        #     sns.lineplot(x=np.arange(0, n_concept+1), y=values_array, label="Dim = " + str(dim))
        #     plt.fill_between(np.arange(0, n_concept+1), values_array - ste_array, values_array + ste_array, alpha=0.2)

        # plt.legend()    
        # plt.savefig("no_wv_hamming_inner_n" + str(n_concept) + ".png")