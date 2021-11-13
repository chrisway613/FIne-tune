# --------------------------------------------------------
# [Configuration] Including data, model, training setting and any others
# Copyright (c) 2021 Moffett.AI
# Licensed under Moffett.AI
# Written by CW
# --------------------------------------------------------

from yacs.config import CfgNode as CN

import os
import yaml


_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 16
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name: squad, squad_v2, etc.
_C.DATA.DATASET = 'squad'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Input sequence maximum length
_C.DATA.MAX_SEQ_LENGTH = 384
# Maximum answer length
_C.DATA.MAX_ANSWER_LENGTH = 30
# Best answer candidates
_C.DATA.N_BEST_ANSWERS = 20
# Overlap length between spans
_C.DATA.DOC_STRIDE = 128
# When input sequence pair, we need specify which one to pad
_C.DATA.PAD_ON_RIGHT = True
# Tensor format: 'pt', 'tf', etc
_C.DATA.TENSOR_FORMAT = 'pt'
# Data format: None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow'
_C.DATA.DATA_FORMAT = 'torch'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'microsoft/deberta-large'
# Model name
_C.MODEL.NAME = 'DeBERTa-large'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 10
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.WARMUP_STEPS = 50
_C.TRAIN.WEIGHT_DECAY = 1e-2
_C.TRAIN.LR = 5e-6  # 16 x V100
_C.TRAIN.MIN_LR = 0.
_C.TRAIN.WARMUP_LR = 0.
_C.TRAIN.LINEAR_SCALED_LR = False
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 1.
# Gradient accumulation steps
_C.TRAIN.ACCUMULATION_STEPS = 1
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'linear'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 1
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'Adam'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-6
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Tag of experiment, overwritten by command line argument
_C.TAG = 'SQUADv1'
# Fixed random seed
_C.SEED = 0
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config: CN, config_file):
    """Update configuration from yaml file."""

    config.defrost()

    with open(config_file, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    
    for cfg in yaml_config.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(config_file), cfg))
    
    print(f'=> merge config from {config_file}\n')

    config.merge_from_file(config_file)
    config.freeze()


def update_config_by_args(config: CN, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)
    
    config.defrost()

    if args.opts:
        config.merge_from_list(args.opts)
    
    # Merge from specific arguments
    if args.tag:
        config.TAG = args.tag
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.linear_scaled_lr:
        config.TRAIN.LINEAR_SCALED_LR = args.linear_scaled_lr
    if args.epoch:
        config.TRAIN.EPOCH = args.epoch
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.output:
        config.OUTPUT = args.output

    # Set local rank for distributed training
    config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    # Output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    
    # Gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        config.TRAIN.LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.MIN_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR *= config.TRAIN.ACCUMULATION_STEPS

    config.freeze()


def get_config(args):
    """
    Get a 'yacs.config.CfgNode' object with default values 
    and merge with command line arguments
    """

    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config_by_args(config, args)

    return config
