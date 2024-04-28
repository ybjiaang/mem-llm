import numpy as np
from scipy.stats import dirichlet
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
from misc import get_filenames

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
        root_dir: str = './save'
        config_file: Optional[str] = None
        exp_id: int=1
        eval: bool = False
        eval_path: Optional[str] = None

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

    if cfg.save_dir is not None:
        outdir = Path(cfg.root_dir) / Path(cfg.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)

    if not cfg.eval:
        ds = Dataset(cfg.data_args)
        cfg.model_args.vocab_size = ds.vocab_size

        model = SelfAttention(cfg.model_args,
                    vocab_size=cfg.model_args.vocab_size)

        # optim
        if cfg.optim_args.use_sgd:
            optimizer = torch.optim.SGD(model.parameters(),
                    lr=cfg.optim_args.learning_rate,
                    weight_decay=cfg.optim_args.weight_decay,
                    momentum=cfg.optim_args.momentum)
        else:
            optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg.optim_args.learning_rate,
                    weight_decay=cfg.optim_args.weight_decay,
                    betas=(0.9, 0.99),
                    eps=1e-8)


        print('Training...')
        training_loss = []
        for i, (x, y) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size,
                                                            num_workers=cfg.num_data_workers)):
            if cfg.max_iters is not None and i >= cfg.max_iters:
                break

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

            optimizer.zero_grad()
            pred = model(x)

            loss = F.cross_entropy(pred[:,-1,:], y[:,-1])

            loss.backward()
            optimizer.step()

            if i % cfg.eval_delta == 0:
                acc = (pred[:,-1,:].argmax(-1) == y[:,-1]).float().mean().item()

                print(i, acc)
                training_loss.append(acc)

        print('Eval...')

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

            print(eval_acc)


        torch.save({'model_state_dict':model.state_dict(), 'ds': ds, 'config': cfg, 'eval_acc': eval_acc, 'training_loss': training_loss}, outdir / Path(str(cfg.exp_id) + ".pth"))

    else:
        filenames = get_filenames(cfg.eval_path)

        for exp_i, filename in enumerate(filenames):

            # create_one_hot_encoding(args)
            checkpoint = torch.load(filename)

            ds = checkpoint['ds']

            cfg = checkpoint['config']

            cfg.model_args.vocab_size = ds.vocab_size

            print(checkpoint['training_loss'])

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

                print(eval_acc)