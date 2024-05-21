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


    eval_list_M = []
    eval_list_Mrand = []
    eval_list_Mgauss = []
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

        Wv = model.attention.wv.weight.detach().numpy()

        unembeddings = model.output.weight.detach().numpy()

        all_vectors = np.array(list(itertools.product([0, 1], repeat=ds.n_concept)))

        
        Wv_lowrank = rank_k_approximation(Wv, k=ds.n_concept)


        M = np.zeros((cfg.model_args.dim, cfg.model_args.dim))
        for vector in all_vectors:
            target_unembed = unembeddings[ds.tokenize(vector)]

            key_unembed = np.zeros(cfg.model_args.dim)
            for i in range(ds.n_concept):
                modefied = vector.copy()
                modefied[i] = 1-vector[i]
                key_unembed += unembeddings[ds.tokenize(modefied)]

            M += np.outer(target_unembed, key_unembed)


        M_random = np.zeros((cfg.model_args.dim, cfg.model_args.dim))
        for vector in all_vectors:
            target_unembed = unembeddings[ds.tokenize(vector)]

            key_unembed = np.zeros(cfg.model_args.dim)
            for i in range(ds.n_concept):
                key_unembed += unembeddings[random.randint(0, 2**ds.n_concept-1)]

            M_random += np.outer(target_unembed, key_unembed)

        M_gauss = np.random.randn(cfg.model_args.dim, cfg.model_args.dim)

        eval_list_M.append(angle_estimate(rank_k_approximation(M, k=ds.n_concept), Wv_lowrank))
        
        eval_list_Mrand.append(angle_estimate(rank_k_approximation(M_random, k=ds.n_concept), Wv_lowrank))

        eval_list_Mgauss.append(angle_estimate(rank_k_approximation(M_gauss, k=ds.n_concept), Wv_lowrank))

    print(np.average(eval_list_M), stats.sem(eval_list_M))
    print(np.average(eval_list_Mrand), stats.sem(eval_list_Mrand))
    print(np.average(eval_list_Mgauss), stats.sem(eval_list_Mgauss))