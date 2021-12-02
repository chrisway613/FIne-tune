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
_C.DATA.DATASET = 'glue'
# ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
_C.DATA.TASK_NAME = None
_C.DATA.LOAD_FROM_CACHE = True
# csv or json
_C.DATA.TRAIN_FILE = None
# csv or json
_C.DATA.VAL_FILE = None
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.TRAIN_BATCH_SIZE = 16
_C.DATA.VAL_BATCH_SIZE = 16
# Official total batch size(per device x num devices per node)
_C.DATA.BASE_BATCH_SIZE = 32 * 8
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Input sequence maximum length
_C.DATA.MAX_SEQ_LENGTH = 256
# Pad all samples to a specific length. Otherwise, dynamic padding is used
_C.DATA.PAD_TO_MAX_SEQ_LENGTH = False

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'microsoft/deberta-large'
# Model classifier dropout rate
_C.MODEL.CLS_DROPOUT = None
# Model name
_C.MODEL.NAME = 'DeBERTa-large'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
_C.MODEL.NO_DECAY_KEYWORDS = ("LayerNorm.weight", "bias")

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 6
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.WARMUP_STEPS = 50
_C.TRAIN.WEIGHT_DECAY = 1e-2
_C.TRAIN.EARLY_STOP = False
_C.TRAIN.MAX_EARLY_STOP_EPOCHS = 20

_C.TRAIN.LR = 1e-5
_C.TRAIN.MIN_LR = 0.
_C.TRAIN.WARMUP_LR = 0.
_C.TRAIN.LINEAR_SCALED_LR = False

# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 1.
# Gradient accumulation steps
_C.TRAIN.GRADIENT_ACCUMULATION_STEPS = 1
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
'''
"linear"
"cosine"
"cosine_with_restarts"
"polynomial"
"constant"
"constant_with_warmup"
'''
_C.TRAIN.LR_SCHEDULER.TYPE = 'linear'
# The number of hard restarts, for 'cosine_with_restarts' scheduler
_C.TRAIN.LR_SCHEDULER.NUM_CYCLES = 3
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 1
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'AdamW'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-6
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.CHILD_TUNING_ADAMW_MODE = 'F'

# Knowledge distillation
_C.TRAIN.KD = CN()
# Whether to use kd
_C.TRAIN.KD.ON = False
# Decide from which Transformer layer we will kd
_C.TRAIN.KD.BEGIN_LAYER = -2
# Kd for logit loss
_C.TRAIN.KD.CLS_LOSS = None
# Kd for Transformer layer loss
_C.TRAIN.KD.REG_LOSS = None
# Teacher state dict
_C.TRAIN.KD.TEACHER_PATH = None

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Tag of experiment, overwritten by command line argument
_C.TAG = 'GLUE'
# Fixed random seed
_C.SEED = 0
# Whether to debug
_C.DEBUG = False
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# Using fast tokenizer by default
_C.USE_SLOW_TOKENIZER = False

# -----------------------------------------------------------------------
# Pruner settings
# -----------------------------------------------------------------------

_C.PRUNE = CN()
_C.PRUNE.PRUNING = False
_C.PRUNE.SPARSITY = 0.9375
_C.PRUNE.DEPLOY_DEVICE = 'none'
_C.PRUNE.GROUP_SIZE = 64
_C.PRUNE.FREQUENCY = 100
_C.PRUNE.FIXED_MASK = None
_C.PRUNE.MASK = None
_C.PRUNE.SPARSE_STEPS = None

# -------------------------------------------------------------------------

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


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
    if args.task_name:
        config.DATA.TASK_NAME = args.task_name

    if args.train_file:
        config.DATA.TRAIN_FILE = args.train_file
    if args.val_file:
        config.DATA.VAL_FILE = args.val_file
    
    if args.max_seq_length:
        config.DATA.MAX_SEQ_LENGTH = args.max_seq_length
    if args.pad_to_max_seq_length:
        config.DATA.PAD_TO_MAX_SEQ_LENGTH = args.pad_to_max_seq_length

    if args.model_type:
        config.MODEL.TYPE = args.model_type
        # microsoft/DeBERTa-base-mnli -> deberta-base-mnli
        config.MODEL.NAME = config.MODEL.TYPE.split('/')[-1].lower()
    if args.cls_dropout:
        config.MODEL.CLS_DROPOUT = args.cls_dropout
    if args.use_slow_tokenizer:
        config.USE_SLOW_TOKENIZER = args.use_slow_tokenizer
    if args.weight_decay is not None:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay

    if args.lr:
        config.TRAIN.LR = args.lr
    if args.linear_scaled_lr:
        config.TRAIN.LINEAR_SCALED_LR = args.linear_scaled_lr
    if args.optimizer:
        config.TRAIN.OPTIMIZER.NAME = args.optimizer
        if args.optimizer == 'child_tuning_adamw' and args.child_tuning_adamw_mode:
            config.TRAIN.OPTIMIZER.CHILD_TUNING_ADAMW_MODE = args.child_tuning_adamw_mode
    if args.lr_scheduler_type:
        config.TRAIN.LR_SCHEDULER.TYPE = args.lr_scheduler_type

    # Gradient accumulation also need to scale the learning rate
    if args.gradient_accumulation_steps:
        config.TRAIN.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    if config.TRAIN.GRADIENT_ACCUMULATION_STEPS > 1:
        config.TRAIN.LR *= config.TRAIN.GRADIENT_ACCUMULATION_STEPS
        config.TRAIN.MIN_LR *= config.TRAIN.GRADIENT_ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR *= config.TRAIN.GRADIENT_ACCUMULATION_STEPS

    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.warmup_steps:
        config.TRAIN.WARMUP_STEPS = args.warmup_steps
    if args.early_stop:
        config.TRAIN.EARLY_STOP = args.early_stop
    if args.max_early_stop_epochs:
        config.TRAIN.MAX_EARLY_STOP_EPOCHS = args.max_early_stop_epochs

    if args.train_batch_size:
        config.DATA.TRAIN_BATCH_SIZE = args.train_batch_size
    if args.val_batch_size:
        config.DATA.VAL_BATCH_SIZE = args.val_batch_size

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.auto_resume:
        config.TRAIN.AUTO_RESUME = args.auto_resume

    # Output folder
    if args.output_dir:
        config.OUTPUT = args.output_dir
    tag = config.DATA.TASK_NAME if config.DATA.TASK_NAME else config.DATA.DATASET
    config.OUTPUT = os.path.join(config.OUTPUT, tag, config.MODEL.NAME.lower())
    
    if args.seed:
        config.SEED = args.seed

    # Set local rank for distributed training
    config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))

    # Pruner settings
    if args.pruning:
        config.PRUNE.PRUNING = args.pruning
    if args.prune_sparsity:
        config.PRUNE.SPARSITY = args.prune_sparsity
    if args.prune_deploy_device:
        config.PRUNE.DEPLOY_DEVICE = args.prune_deploy_device
    if args.prune_group_size:
        config.PRUNE.GROUP_SIZE = args.prune_group_size
    if args.prune_frequency:
        config.PRUNE.FREQUENCY = args.prune_frequency
    if args.fixed_mask:
        config.PRUNE.FIXED_MASK = args.fixed_mask
    if args.mask:
        config.PRUNE.MASK = args.mask
    if args.sparse_steps:
        config.PRUNE.SPARSE_STEPS = args.sparse_steps
    
    if args.kd_on:
        config.TRAIN.KD.ON = args.kd_on
        if args.kd_cls_loss:
            config.TRAIN.KD.CLS_LOSS = args.kd_cls_loss
        if args.kd_reg_loss:
            config.TRAIN.KD.REG_LOSS = args.kd_reg_loss
        if args.teacher_path:
            config.TRAIN.KD.TEACHER_PATH = args.teacher_path

    if args.debug:
        config.DEBUG = True

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
