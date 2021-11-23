# --------------------------------------------------------
# [Random Seed] Usually we should fix the random seed in order to reproduce the lab result.
# Copyright (c) 2021 Moffett.AI
# Licensed under Moffett.AI
# Written by CW
# --------------------------------------------------------

import torch
import random
import numpy as np

from functools import partial


def setup_seed(seed, deterministic=False):
    """
    Fix random seed
    Note: random, numpy, torch, cudnn should be set respectively.
    """

    # Random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch cpu
    torch.manual_seed(seed)
    # Torch gpu
    if torch.cuda.is_available():
        # Apply to all gpus
        torch.cuda.manual_seed_all(seed)
        # Cudnn
        # Do not search for the fastest convolution algorithm
        torch.backends.cudnn.benchmark = False
        # For stochastic convolution implementation
        torch.backends.cudnn.deterministic = True

        if deterministic:
            # 固定住一些本身带有随机性的算法，如 Dropout，但如果有些算法没有对应的实现，
            # 则会报错：
            #  RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation
            # 另外，使用这种做法需要设置环境变量：
            # this operation is not deterministic because it 
            # uses CuBLAS and you have CUDA >= 10.2. 
            # To enable deterministic behavior in this case, 
            # you must set an environment variable before running your PyTorch application: 
            # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. 
            # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            torch.use_deterministic_algorithms(True)


def reseed_worker(worker_id, num_workers, seed, rank=0):
    """Reseed dataloader workers"""
    worker_seed = worker_id + seed + num_workers * rank

    np.random.seed(worker_seed) 
    random.seed(worker_seed) 


def reseed_workers_fn(num_workers, seed, rank=0):
    """Do reseed if there are multiple dataloader workers with multi processes"""
    return partial(reseed_worker, num_workers=num_workers, seed=seed, rank=rank) \
        if num_workers > 0 else None
