# --------------------------------------------------------
# [Distributed Training Pipeline]
# Copyright (c) 2021 Moffett.AI
# Licensed under Moffett.AI
# Written by CW
# --------------------------------------------------------

"""
    Finetuning a 🤗 Transformers model for sequence classification on GLUE.
    Run tips:
    i.   Run: accelerate config
    ii.  Reply the questions in order to setup your configuration
    iii. Run this script like:

    TOKENIZERS_PARALLELISM=true accelerate launch run_glue.py --task_name [TASK_NAME] \
        --model_type [MODEL_NAME] \
        --output_dir [OUTPUT_DIR] --train_batch_size [TRAIN_BATCH_SIZE] --val_batch_size [VAL_BATCH_SIZE] \
        --pad_to_max_seq_length --linear_scaled_lr --weight_decay [WEIGHT_DECAY] --pruning \
        --prune_frequency [PRUNE_FREQUENCY] --kd_on..
"""

import os
import time
import random
import argparse
import datetime
import datasets
import transformers
import torch.nn as nn

from tqdm import tqdm

from torch.optim import optimizer
from torch.utils.data import DataLoader

from datasets import load_metric
from accelerate import Accelerator, DistributedType

from transformers import (
    # AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    SchedulerType,
    default_data_collator,
    DataCollatorWithPadding,
)
from transformers.utils.versions import require_version


require_version("datasets>=1.8.0", 
                "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from utils.logger import Logger
from utils.seed import setup_seed, reseed_workers_fn
from utils.dist import kill_all_process
from utils.misc import auto_resume_helper, load_checkpoint, save_checkpoint

from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler

from data.glue.load import load_data
from data.glue.process import preprocess_data

from trainer_accelerate import Trainer
from configs.glue.cfg import get_config, TASK_TO_KEYS

from pruner import Prune
from loss import loss_dict


def parse_args():
    """Parse command line arguments & update default configurations"""

    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(TASK_TO_KEYS.keys()),
    )
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--val_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        )
    )
    parser.add_argument(
        "--pad_to_max_seq_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--cls_dropout",
        type=float,
        help='model classifier dropout rate'
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument('--linear_scaled_lr', action='store_true',
                        help='linear scale the learning rate according to total batch size')
    
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay to use.")
    
    parser.add_argument("--epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, help="Number of steps for the warmup in the lr scheduler."
    )
    
    parser.add_argument("--seed", type=int, help="A seed for reproducible training.")
    parser.add_argument('--resume', type=str, help='path to the checkpoint should be resume from')
    parser.add_argument('--auto_resume', action='store_true', help='whether to auto resume from history')
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to store the final model.")

    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, 
                        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    
    parser.add_argument("--opts", default=None, nargs='+', help="Modify config options by adding 'KEY VALUE' pairs")

    # Prune arguments
    parser.add_argument('--pruning', action='store_true', help='whether to prune')
    parser.add_argument('--aug_train', action='store_true')
    parser.add_argument('--pred_distill', action='store_true')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--prune_sparsity',type=float, help='sparsity rate')
    parser.add_argument('--prune_deploy_device',type=str,
                        help='also known as balance. options none, fix=asic, fpga')
    parser.add_argument('--prune_group_size',type=int, help='also known as bank_size')
    parser.add_argument('--prune_frequency',type=int, help='also known as bank_size')
    parser.add_argument('--fixed_mask', type=str, help="Fixed mask path.")
    parser.add_argument('--mask', type=str, help="mask path")

    parser.add_argument('--kd_on', action='store_true', help='whether to use knowledge distillation')
    parser.add_argument('--kd_cls_loss', default='soft_ce', help='kd loss for output logits')
    parser.add_argument('--kd_reg_loss', default='mse', help='kd loss for Transformer layers')

    parser.add_argument('--debug', action='store_true', help='whether to debug')

    args, _ = parser.parse_known_args()
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.val_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.val_file is not None:
            extension = args.val_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    # Update the default configuration by command line arguments
    cfg = get_config(args)
    return args, cfg


if __name__ == '__main__':
    '''i. Parse arguments & Update configs'''
    _, cfg = parse_args()

    '''ii. Set logger'''
    now = datetime.datetime.now().strftime("%m-%d-%H-%M")
    log_dir = os.path.join(cfg.OUTPUT, now)
    logger = Logger(log_dir, dist_rank=cfg.LOCAL_RANK, name=cfg.MODEL.NAME)
    logger.info(f"=> Log info to file: '{logger.log_file}'")

    '''iii. Initialize the accelerator'''
    # We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # We only want one process per machine to log things on the screen.
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info(f"\nProcess id: {os.getpid()}\n{accelerator.state}")

    '''iv. Fix random seed'''
    # rank_seed = cfg.SEED + cfg.LOCAL_RANK
    # rank_seed = cfg.SEED + accelerator.local_process_index
    # setup_seed(rank_seed)
    # logger.info(f"=> Rank random seed={rank_seed}\n")
    setup_seed(cfg.SEED)
    logger.info(f"=> Random seed: {cfg.SEED}\n")

    accelerator.wait_for_everyone()

    '''v. Load dataset'''
    s = time.time()
    task_name = cfg.DATA.TASK_NAME if (cfg.DATA.TRAIN_FILE is None or \
        cfg.DATA.VAL_FILE is None) else None

    data, label_list, num_labels, is_regression, metric_computor = load_data(
        task_name=task_name, train_file=cfg.DATA.TRAIN_FILE, val_file=cfg.DATA.VAL_FILE
    )
    if metric_computor is None:
        if cfg.DATA.TASK_NAME is not None:
            metric_computor = load_metric('glue', cfg.DATA.TASK_NAME)
        else:
            metric_computor = load_metric('accuracy')

    used = time.time() - s
    logger.info(f"\n[Dataset]\n{data}\nLoad data takes time:{datetime.timedelta(seconds=used)}\n")

    '''vi. Build model and tokenizer'''
    # In distributed training, the 'from_pretrained' methods guarantee that 
    # only one local process can concurrently download model & vocab.
    s = time.time()

    # For custom model config
    # from transformers.models.deberta.configuration_deberta import DebertaConfig

    # model_config_dict = {
    #     "model_type": "deberta",
    #     "attention_probs_dropout_prob": 0.1,
    #     "hidden_act": "gelu",
    #     "hidden_dropout_prob": 0.1,
    #     "hidden_size": 1024,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 4096,
    #     "max_position_embeddings": 512,
    #     "relative_attention": False,
    #     "pos_att_type": None,
    #     "layer_norm_eps": 1e-7,
    #     "max_relative_positions": -1,
    #     "position_biased_input": False,
    #     "num_attention_heads": 16,
    #     "num_hidden_layers": 24,
    #     "type_vocab_size": 0,
    #     "vocab_size": 50265
    # }
    # model_config = DebertaConfig(**model_config_dict)

    # from transformers.models.deberta.modeling_deberta import DebertaForSequenceClassification

    # model = DebertaForSequenceClassification._from_config(model_config)
    
    # auto_config = AutoConfig.from_pretrained(cfg.MODEL.TYPE, 
    #                                          num_labels=num_labels, finetuning_task=cfg.DATA.TASK_NAME)
    # logger.info(f"\n=> auto_config type:{type(auto_config)}\n content:{auto_config}\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL.TYPE,
        # TODO: may occur error when use another one down-stream task pretrained weight
        # config=auto_config
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.TYPE, use_fast=not cfg.USE_SLOW_TOKENIZER)
    used = time.time() - s

    # Alter the classifier dropout rate
    if getattr(model, 'dropout', None) and cfg.MODEL.CLS_DROPOUT is not None:
        logger.info(f"\nSet model classifier dropout rate from "
                    f"{model.dropout.drop_prob} to {cfg.MODEL.CLS_DROPOUT}\n")
        model.dropout.drop_prob = cfg.MODEL.CLS_DROPOUT
        # Also change the value in model config
        if getattr(model.config, "cls_dropout", None):
            model.config.cls_dropout = cfg.MODEL.CLS_DROPOUT

    # Change the label mapping of model
    if getattr(model, 'classifier', None) and model.classifier.out_features != num_labels:
        logger.warning(f"\nModel classifier does not match the dataset category:\n"
                       f"model output features: {model.classifier.out_features}, dataset number of labels: {num_labels}\n"
                       f"Now, change to consistent with dataset.\n")
        
        if getattr(model, 'num_labels', None) != num_labels:
            model.num_labels = num_labels
        
        in_features, bias = model.classifier.in_features, model.classifier.bias
        model.classifier = nn.Linear(in_features, num_labels, bias=bias is not None)

    logger.info(f"=> Build model '{cfg.MODEL.NAME}' from pretrained '{cfg.MODEL.TYPE}'")
    logger.info(f"{str(model)}\n")
    logger.info(f"=> Load model(w tokenizer) takes time: {datetime.timedelta(seconds=used)}\n")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"=> Number of model params: {n_parameters}")

    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"=> Number of FLOPs: {flops / 1e9}G\n")
    
    teacher = kd_logit_loss = kd_layer_loss = None
    if cfg.TRAIN.KD.ON:
        teacher = AutoModelForSequenceClassification.from_pretrained(
            cfg.MODEL.TYPE,
            # TODO: may occure error when use another one down-stream task pretrained weight
            # config=auto_config
        )
        # Pay attention to set the teacher to eval model
        teacher.eval()

        kd_logit_loss = loss_dict.get(cfg.TRAIN.KD.CLS_LOSS)
        kd_layer_loss = loss_dict.get(cfg.TRAIN.KD.REG_LOSS)

        # TODO: how to output the name of a variable
        logger.info(f"\nKD mode: ON\nTeacher model: {teacher.__class__.__name__}\n"
                    f"Output logit loss: {kd_logit_loss.__name__}\n"
                    f"Transformer layer loss: {kd_layer_loss.__name__}\n")

    '''vii. Preprocess dataset then feed in dataloader'''
    s = time.time()
    # processed_data = preprocess_data(
    #     data, model, tokenizer, auto_config, num_labels, 
    #     label_list, is_regression, logger, cfg, accelerator
    # )
    processed_data = preprocess_data(
        data, model, tokenizer, num_labels, 
        label_list, is_regression, logger, cfg, accelerator,
        task_name=task_name
    )
    used = time.time() - s
    logger.info(f"=> Process data takes time:{datetime.timedelta(seconds=used)}\n")

    train_data = processed_data['train']
    val_data = processed_data['validation_matched' \
        if cfg.DATA.TASK_NAME == 'mnli' else 'validation']
    
    if cfg.DEBUG:
        num_train_samples = cfg.DATA.TRAIN_BATCH_SIZE * accelerator.num_processes
        num_val_samples = cfg.DATA.VAL_BATCH_SIZE * accelerator.num_processes

        train_data = train_data.select(range(num_train_samples))
        val_data = val_data.select(range(num_val_samples))

        logger.info(f"=> Debug mode on! {num_train_samples} train samples & {num_val_samples} val samples selected\n")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_data)), 2):
        logger.info(f"=> Sample {index} of the training set: {train_data[index]}.")
    
    # DataLoaders creation:
    if cfg.DATA.PAD_TO_MAX_SEQ_LENGTH:
        # If padding was already done ot max length, 
        # we use the default data collator that will just convert everything to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    s = time.time()
    train_dataloader = DataLoader(
        train_data, batch_size=cfg.DATA.TRAIN_BATCH_SIZE,
        shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=data_collator, pin_memory=cfg.DATA.PIN_MEMORY,
        # Set 'worker_init_fn' for randomness of multi-process
        # Random seed of different sub process in different rank is different
        worker_init_fn=reseed_workers_fn(cfg.DATA.NUM_WORKERS, cfg.SEED, rank=cfg.LOCAL_RANK)
    )
    val_dataloader = DataLoader(
        val_data, batch_size=cfg.DATA.VAL_BATCH_SIZE, 
        num_workers=cfg.DATA.NUM_WORKERS, pin_memory=cfg.DATA.PIN_MEMORY,
        collate_fn=data_collator
    )
    used = time.time() - s
    logger.info(f"=> Dataloader takes time:{datetime.timedelta(used)}\n")

    '''viii. Build optimizer & lr_scheduler'''
    # Linear scale the learning rate according to total batch size
    if cfg.TRAIN.LINEAR_SCALED_LR:
        cfg.defrost()

        # 256=16(batch size) x 16(v100)
        scaled = cfg.DATA.TRAIN_BATCH_SIZE * accelerator.num_processes / cfg.DATA.BASE_BATCH_SIZE
        cfg.TRAIN.LR *= scaled
        cfg.TRAIN.MIN_LR *= scaled
        cfg.TRAIN.WARMUP_LR *= scaled
        # TODO: test this by resulting
        # cfg.TRAIN.WARMUP_STEPS = int(cfg.TRAIN.WARMUP_STEPS / scaled)

        cfg.freeze()
    optimizer = build_optimizer(model, cfg)

    # In multi-process, each will get its own individual data
    # (cuz the length of dataloader would be shorter than the original)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    if teacher is not None:
        teacher = accelerator.prepare_model(teacher)

    # Note: this 'num_train_steps' considers gradient accumulation
    # this is the frequency that lr scheduler updates
    lr_scheduler, num_train_steps = build_lr_scheduler(optimizer, cfg, len(train_dataloader))

    '''ix. Training preparation'''
    best_val_results = {'accuracy': 0., 'f1': 0., 
                        'pearson': 0., 'spearmanr': 0., 'matthews_correlation': 0.}

    if cfg.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(cfg.OUTPUT)
        if resume_file:
            if cfg.MODEL.RESUME:
                logger.warning(
                    f"=> Auto-resume changing resume file from '{cfg.MODEL.RESUME}' to '{resume_file}'"
                )

            cfg.defrost()
            cfg.MODEL.RESUME = resume_file
            cfg.freeze()

            logger.info(f"=> Auto resuming from '{resume_file}'..")
            # Dict: metric type -> metric value
            best_val_results = load_checkpoint(
                accelerator.unwrap_model(model), accelerator.unwrap_model(optimizer), 
                lr_scheduler, cfg, logger
            )
            logger.info(f"=> Auto resume done!\n")
        else:
            logger.warning(f"=> No checkpoint found in '{cfg.OUTPUT}', ignoring auto resume\n")
    
    # Log config & training information
    logger.info(f"\n[Config]\n{cfg.dump()}\n")
    total_batch_size = cfg.DATA.TRAIN_BATCH_SIZE * accelerator.num_processes * cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS
    logger.info("***** Start Training *****")
    logger.info(f"  Num train examples(all devices) = {len(train_data)}")
    logger.info(f"  Num val examples(all devices) = {len(val_data)}")
    logger.info(f"  Num Epochs = {cfg.TRAIN.EPOCHS}")
    logger.info(f"  train batch size per device = {cfg.DATA.TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total train batch size \
        (batch size per device x num devices x gradient accumulation steps) = {total_batch_size}\n")

    '''x. Pruner setting'''
    if cfg.PRUNE.PRUNING:
        prune_dict = {}
        for name, _ in accelerator.unwrap_model(model).named_parameters():
            if 'deberta' in cfg.MODEL.NAME.lower():
                # FFN
                if ('intermediate.dense.weight' in name or 'output.dense.weight' in name) \
                    and ('attention' not in name):
                    prune_dict[name] = cfg.PRUNE.SPARSITY
                # Attention
                if 'attention.self.in_proj.weight' in name or 'attention.self.pos_proj.weight' in name \
                    or 'attention.self.pos_q_proj.weight' in name or 'attention.output.dense.weight' in name:
                    prune_dict[name] = cfg.PRUNE.SPARSITY
            else:
                pass
        logger.info(f"=> \n[Prune Dict]\n{prune_dict}\n")
        
        pruner = Prune(
            model=accelerator.unwrap_model(model), 
            pretrain_step=0,
            sparse_step=num_train_steps,
            frequency=cfg.PRUNE.FREQUENCY,
            prune_dict=prune_dict,
            restore_sparsity=False,
            fix_sparsity=False,
            prune_device='default',
            deploy_device=cfg.PRUNE.DEPLOY_DEVICE,
            group_size=cfg.PRUNE.GROUP_SIZE,
            fixed_mask=cfg.PRUNE.FIXED_MASK,
            mask=cfg.PRUNE.MASK
        )
    else:
        pruner = None

    '''xi. Training'''
    logger.info(f"=> Start training\n")

    best_checkpoint_dir = os.path.join(log_dir, 'best')
    os.makedirs(best_checkpoint_dir, exist_ok=True)

    # Only show the progress bar once on each machine.
    # Note: this 'num_train_steps' considers gradient accumulation
    # this is the frequency that lr scheduler updates
    progress_bar = tqdm(range(num_train_steps), disable=not accelerator.is_local_main_process)

    accelerator.wait_for_everyone()

    begin = time.time()
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        Trainer.train(accelerator, model, train_dataloader,
                      optimizer, lr_scheduler, cfg, logger, epoch, progress_bar,
                      pruner=pruner, teacher=teacher, kd_cls_loss=kd_logit_loss, kd_reg_loss=kd_layer_loss)

        if accelerator.is_local_main_process and (not epoch % cfg.SAVE_FREQ or epoch == cfg.TRAIN.EPOCHS - 1):
            checkpoint = save_checkpoint(
                log_dir, accelerator.unwrap_model(model), 
                accelerator.unwrap_model(optimizer), lr_scheduler, epoch, cfg, best_val_results,
                tokenizer=tokenizer, accelerator=accelerator
            )

            logger.info(f"=> Checkpoint '{checkpoint}' saved\n")

        # Eval
        accelerator.wait_for_everyone()
        val_results = Trainer.val(accelerator, model, val_dataloader, cfg, logger, 
                                  epoch, metric_computor, is_regression)

        if cfg.DATA.TASK_NAME.lower() in ('mnli', 'qnli', 'rte', 'sst-2', 'wnli'):
            if val_results['accuracy'] > best_val_results['accuracy'] and accelerator.is_local_main_process:
                best_val_results['accuracy'] = val_results['accuracy']
                best_checkpoint = save_checkpoint(
                    best_checkpoint_dir, accelerator.unwrap_model(model), 
                    accelerator.unwrap_model(optimizer), lr_scheduler, 
                    epoch, cfg, best_val_results, tokenizer=tokenizer, accelerator=accelerator
                )

                logger.info(f"=> Best checkpoint '{best_checkpoint}' saved\n")
        else:
            pass

    total = time.time() - begin
    total_str = str(datetime.timedelta(seconds=total))
    logger.info(f"=> Training finished! time used: {total_str}\n")

    # Final evaluation on mismatched validation set
    if cfg.DATA.TASK_NAME.lower() == "mnli":
        # TODO: make this elegant
        eval_size = cfg.DATA.VAL_BATCH_SIZE * accelerator.num_processes
        eval_dataset = processed_data["validation_mismatched"].select(range(eval_size))
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=cfg.DATA.VAL_BATCH_SIZE
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)
        logger.info(f"=> Evaluate on MNLI-MISMATCHED validation set({eval_size} samples)")

        Trainer.val(accelerator, model, eval_dataloader, cfg, 
                    logger, cfg.TRAIN.EPOCHS, metric_computor, False)

    # Note: this is for EF scheduler, it is required
    logger.info("Success")

    # Release all references to the internal objects stored and call the garbage collector
    accelerator.free_memory()
    if accelerator.distributed_type == DistributedType.MULTI_GPU:
        # Destroy all processes, and de-initialize the distributed package
        kill_all_process()
