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
import pandas as pd

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

    hamming_cos_dict = {}
    
    acc_1_list = []
    acc_2_list = []
    stderr_1_list = []
    stderr_2_list = []
    for prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        eval_list_1 = []
        eval_list_2 = []
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
            for i, (x, y) in enumerate(iterate_batches(ds, batch_size=2048,
                                                                num_workers=cfg.num_data_workers)):
                if cfg.max_iters is not None and i >= cfg.max_iters:
                    break

                y_1 = y[:1024,-1]
                y_2 = y[1024:,-1]
                x_1 = x[:1024,:]
                x_2 = x[1024:,:]

                matrix = np.random.binomial(1, prob, size=(x_1.shape[0], x_1.shape[1]))

                x = x_1 * (1-matrix) + x_2 * (matrix)


                x = torch.from_numpy(x)
                y_1 = torch.from_numpy(y_1)
                y_2 = torch.from_numpy(y_2)

                pred = model(x)

                eval_acc_1 = (pred[:,-1,:].argmax(-1) == y_1).float().mean().item()
                eval_acc_2 = (pred[:,-1,:].argmax(-1) == y_2).float().mean().item()

                eval_list_1.append(eval_acc_1)
                eval_list_2.append(eval_acc_2)


        # print(eval_list)
        acc_1 = np.average(eval_list_1)
        stderr_1 = stats.sem(eval_list_1)

        acc_2 = np.average(eval_list_2)
        stderr_2 = stats.sem(eval_list_2)

        acc_1_list.append(acc_1)
        acc_2_list.append(acc_2)
        stderr_1_list.append(stderr_1)
        stderr_2_list.append(stderr_2)


data = {
    'Category': ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
    'Target True': acc_1_list,
    'Target False': acc_2_list,
    
}

frozen_stderr = np.array(stderr_1_list)
training_stderr= np.array(stderr_2_list)


# Convert data to a DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to long format
df_melted = df.melt(id_vars='Category', var_name='Group', value_name='Value')

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x='Category', y='Value', hue='Group', data=df_melted)
plt.errorbar(x=[index - 0.2 for index in range(len(df_melted['Category'].unique()))], y=df['Target True'], yerr=frozen_stderr, fmt='none', color='black', capsize=5)
plt.errorbar(x=[index + 0.2 for index in range(len(df_melted['Category'].unique()))], y=df['Target False'], yerr=training_stderr, fmt='none', color='black', capsize=5)
plt.xlabel('Mixing Rate', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

# Adjust legend
plt.legend(title='', loc='lower left')
plt.savefig("confusion_bar_l" + str(ds.n_concept) + ".png")

