# --------------------------------------------------------
# [Distributed Training Pipeline]
# Copyright (c) 2021 Moffett.AI
# Licensed under Moffett.AI
# Written by CW
# --------------------------------------------------------

"""
    i). torch.distributed.launch
    CUBLAS_WORKSPACE_CONFIG=:16:8 TOKENIZERS_PARALLELISM=true \
    CUDA_VISIBLE_DEVICES=0,1,2,.. python -m torch.distributed.launch \
        --nproc_per_node [num_procs] train_dist.py [Other parameters]
    
    ii). torchrun
    TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0,1,2,.. \
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS train_dist.py (--arg1 ...)
"""

import time
import torch
import datetime
import argparse

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering

import os


# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# os.environ['TOKENIZERS_PARALLELISM'] = True 

import sys

# Add project main directory to system path for module reference
# This should be called before you import your self-defined module
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from configs.squad.cfg import get_config

from utils.logger import Logger
from utils.seed import setup_seed
from utils.misc import auto_resume_helper, load_checkpoint, save_checkpoint
from utils.dist import get_rank, get_world_size, init_process_group, is_master, distributed_master_first

from data.squad.load import load_data
from data.squad.process import generate_features

from trainer import Trainer
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler


def parse_options():
    """Parse command line arguments & update the default configuration."""

    parser = argparse.ArgumentParser(description='DeBERTa Fine-tuning script')

    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')

    parser.add_argument('--epoch', type=int, help='training epochs')
    parser.add_argument('--lr', type=float, help='base learning rate')
    parser.add_argument('--linear_scaled_lr', action='store_true',
                        help='linear scale the learning rate according to total batch size')
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")

    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--resume', type=str, help='path to the checkpoint should be resume from')
    parser.add_argument('--output', default='outputs', type=str, metavar='PATH',
                        help='root of output folder, the full path is <outputs>/<model_name>/<tag> (default: outputs)')

    parser.add_argument("--opts", default=None, nargs='+', help="Modify config options by adding 'KEY VALUE' pairs")
    # 'RANK' and 'WORLD_SIZE' are assigned automatically by torchrun:
    # int(os.environ["LOCAL_RANK"]) int(os.environ["WORLD_SIZE"])
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, _ = parser.parse_known_args()
    # Update the default configuration by command line arguments
    config = get_config(args)

    return args, config


if __name__ == '__main__':
    '''i. Parse arguments & Update configs'''
    _, config = parse_options()

    '''ii. Set logger'''
    now = datetime.datetime.now().strftime("%m-%d-%H-%M")
    log_dir = os.path.join(config.OUTPUT, now)
    logger = Logger(log_dir, dist_rank=config.LOCAL_RANK, name=config.MODEL.NAME)
    logger.info(f"=> Log info to file: '{logger.log_file}'")

    '''iii. Distributed setting'''
    torch.cuda.set_device(config.LOCAL_RANK)
    init_process_group()
    device = torch.device('cuda')
    logger.info(f"=> [rank]/[world_size]=[{get_rank()}]/[{get_world_size()}] gpu:{torch.cuda.current_device()}")

    '''iv. Fix random seed'''
    rank_seed = config.SEED + get_rank()
    setup_seed(rank_seed)
    logger.info(f"=> Rank random seed={rank_seed}\n")

    '''v. Data processing'''
    # Load
    data, metric_computor = load_data(config.DATA.DATASET)
    logger.info(f"\n[Dataset]\n{data}\n")

    # TODO: delete this debugging
    # train_data = data['train'].select(range(config.DATA.BATCH_SIZE * 4))
    # val_data_raw = data['validation'].select(range(config.DATA.BATCH_SIZE * 4))

    # Tokenize
    with distributed_master_first():
        tokenized_data = data.map(generate_features(), 
                                batched=True, remove_columns=data['train'].column_names)
    tokenized_train_data, tokenized_val_data = tokenized_data['train'], tokenized_data['validation']

    # TODO: remove this debugging
    # tokenized_train_data = train_data.map(generate_features(), batched=True, remove_columns=train_data.column_names)
    # tokenized_val_data = val_data_raw.map(generate_features(), batched=True, remove_columns=val_data_raw.column_names)
    
    # Batch Data
    tokenized_train_data.set_format(config.DATA.DATA_FORMAT)
    train_data = tokenized_train_data.shuffle(rank_seed)

    tokenized_val_data.set_format(config.DATA.DATA_FORMAT)
    # Validation set do not need to be shuffled
    val_data = tokenized_val_data

    train_dist_sampler = DistributedSampler(train_data)
    # Note: please explicitly set 'shuffle=False' cuz default is 'shuffle=True'
    val_dist_sampler = DistributedSampler(val_data, shuffle=False)

    train_dataloader = DataLoader(train_data, batch_size=config.DATA.BATCH_SIZE, sampler=train_dist_sampler,
                                  num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY)
    val_dataloader = DataLoader(val_data, batch_size=config.DATA.BATCH_SIZE, sampler=val_dist_sampler,
                                num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY)

    logger.info(f"=> Total {len(train_dataloader)} batch of {len(train_data) // get_world_size()} train samples")
    logger.info(f"=> Total {len(val_dataloader)} batch of {len(val_data) // get_world_size()} val samples\n")

    # Generate features for evaluation
    # Get each rank's subset of data
    rank_val_data_raw = data['validation'].select(
        range(config.LOCAL_RANK, len(data['validation']), get_world_size()))
    val_features = rank_val_data_raw.map(
        generate_features(mode='val'),
        batched=True, remove_columns=rank_val_data_raw.column_names
    )
    # TODO: remove this debugging intent
    # rank_val_data_raw = val_data_raw.select(range(config.LOCAL_RANK, len(val_data_raw), get_world_size()))
    # val_features = rank_val_data_raw.map(
    #     generate_features(mode='val'),
    #     batched=True, remove_columns=rank_val_data_raw.column_names
    # )
    val_features.set_format(type=val_features.format['type'],
                            columns=list(val_features.features.keys()))

    '''vi. Build model'''
    model = AutoModelForQuestionAnswering.from_pretrained(config.MODEL.TYPE)
    model.to(device)
    model = DDP(model, device_ids=[config.LOCAL_RANK])
    model_wo_ddp = model.module

    logger.info(f"=> Build model '{config.MODEL.NAME} from pretrained '{config.MODEL.TYPE}'")
    logger.info(f"{str(model_wo_ddp)}\n")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"=> number of model params: {n_parameters}")

    if hasattr(model_wo_ddp, 'flops'):
        flops = model_wo_ddp.flops()
        logger.info(f"=> number of FLOPs: {flops / 1e9}G\n")

    '''vii. Determine lr, may be linear scaled'''
    # Linear scale the learning rate according to total batch size
    if config.TRAIN.LINEAR_SCALED_LR:
        config.defrost()

        # 256=16(batch size) x 16(v100)
        scaled = config.DATA.BATCH_SIZE * get_world_size() / 256
        config.TRAIN.LR *= scaled
        config.TRAIN.MIN_LR *= scaled
        config.TRAIN.WARMUP_LR *= scaled

        config.freeze()

    '''viii. Build optimizer & lr scheduler'''
    optimizer = build_optimizer(model_wo_ddp, config)
    lr_scheduler = build_lr_scheduler(optimizer, config, len(train_dataloader))

    best_f1 = best_em = 0.
    best_checkpoint_dir = os.path.join(log_dir, 'best')

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"=> Auto-resume changing resume file from '{config.MODEL.RESUME}' to '{resume_file}'"
                )

            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()

            logger.info(f"=> Auto resuming from '{resume_file}'..")
            best_metrics = load_checkpoint(model_wo_ddp, optimizer, lr_scheduler, config, logger)
            best_f1, best_em = best_metrics['f1'], best_metrics['em']
            logger.info(f"=> Auto resume done!\n")
        else:
            logger.warning(f"=> No checkpoint found in '{config.OUTPUT}', ignoring auto resume\n")

    # Log config
    logger.info(f"\n[Config]\n{config.dump()}\n")

    '''ix. Training'''
    logger.info(f"=> Start training\n")

    train_steps = len(train_dataloader) * config.TRAIN.EPOCHS
    progress_bar = tqdm(range(train_steps), disable=not is_master())

    begin = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # Make each epoch shuffle
        train_dataloader.sampler.set_epoch(epoch)
        Trainer.train(model, train_dataloader, optimizer, lr_scheduler, 
                      config, logger, epoch, progress_bar, device)

        if is_master() and (not epoch % config.SAVE_FREQ or epoch == config.TRAIN.EPOCHS - 1):
            checkpoint = save_checkpoint(
                log_dir, model_wo_ddp, 
                optimizer, lr_scheduler, epoch, config, best_metrics
            )
            logger.info(f"=> checkpoint '{checkpoint}' saved\n")
        
        f1, em = Trainer.val(model, val_dataloader, rank_val_data_raw,
                             val_features, metric_computor, config, logger, epoch, device)
        prev_f1, prev_em = best_metrics['f1'], best_metrics['em']

        if em > prev_em:
            best_metrics['em'] = em
            logger.info(f"=> Gain best EM: {em:.2f}")
        if f1 > prev_f1:
            best_metrics['f1'] = f1
            logger.info(f"=> Gain best F1: {f1:.2f}")

        if f1 + em > prev_f1 + prev_em and is_master():
            best_checkpoint = save_checkpoint(
                best_checkpoint_dir, model_wo_ddp, 
                optimizer, lr_scheduler, epoch, config, best_metrics
            )
            logger.info(f"=> best checkpoint '{best_checkpoint}' saved\n")

        # TODO: remove this debugging intent
        # break

    total = time.time() - begin
    total_str = str(datetime.timedelta(seconds=total))
    logger.info(f"=> Training finished! time used: {total_str}\n")
