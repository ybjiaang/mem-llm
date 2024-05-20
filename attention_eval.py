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


    filenames = get_filenames(cfg.eval_path)

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

        
        vocab_sequence = np.arange(effective_vocab_size).reshape(1, effective_vocab_size)

        vocab_embeddings = model.tok_embeddings(torch.from_numpy(vocab_sequence))

        heat_map += model.attention.get_attention_heatmap(vocab_embeddings)[0,0,:,:].detach().numpy()
        

    
    heat_map /= len(filenames)
    print(np.average(eval_list))
    print(stats.sem(eval_list))

    plt.imshow(heat_map.T, cmap='viridis', interpolation='nearest')
    plt.xlabel('Key Tokens', fontsize=20)
    plt.ylabel('Query Tokens', fontsize=20)
    
    plt.colorbar()
    plt.savefig("heat_map_n" + str(ds.n_concept)+".png")

    