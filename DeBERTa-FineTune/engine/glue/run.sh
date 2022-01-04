#!/bin/bash

# Task & Model
TASK=rte
MODEL=deberta-base-mnli

# Train
EPOCHS=10
VAL_BS=16
TRAIN_BS=16
WARMUP_STEPS=20
RESUME=outputs/rte/deberta-base-mnli/ep50-lr3e-05-pruneTrue-pfreq400-psteps180800-sparsity0.9375/12-26-10-48/epoch44-acc0.81.pth

# Optimization
LR=2e-5
OPTIMIZER=adamw
SCHEDULER=constant_linear
WEIGHT_DECAY=1e-2

# Data
TRAIN_FILE=/home/user/weicai/datasets/rte/aug/train_aug.csv
VAL_FILE=/home/user/weicai/datasets/rte/aug/dev.csv

# Prune
PRUNE_FREQ=1
SPARSE_STEPS=1
SPARSITY=0.5
PRUNE_DEPLOY_DEVICE=none

# Kd
# TEACHER=/home/user/weicai/Fine-tune/DeBERTa-FineTune/engine/glue/aug_dense_84.38.pth
TEACHER=base-rte-4x3090-ep6-bs16-lr2e-5-acc85.3.pth
# TEACHER=base-mrpc-8x3090-ep6-bs32-lr2e-5-avg88.5-acc86.7-f190.3.pth

# Log
NOHUP_OUTPUT=outputs/$TASK/$MODEL/ep$EPOCHS-lr$LR-freq$PRUNE_FREQ-steps$SPARSE_STEPS.log

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true MPLBACKEND='Agg' \
  nohup accelerate launch run_glue.py --task_name $TASK --model_type microsoft/$MODEL \
  --train_batch_size $TRAIN_BS \
  --val_batch_size $VAL_BS \
  --epochs $EPOCHS --warmup_steps $WARMUP_STEPS \
  --lr $LR --optimizer $OPTIMIZER --lr_scheduler_type $SCHEDULER \
  --max_seq_length 320 --pad_to_max_seq_length \
  --cls_dropout 0.2 --weight_decay $WEIGHT_DECAY \
  --pruning --prune_frequency $PRUNE_FREQ --prune_sparsity $SPARSITY \
  --sparse_steps $SPARSE_STEPS --prune_deploy_device $PRUNE_DEPLOY_DEVICE \
  --kd_on --teacher_init --teacher_path $TEACHER  >> $NOHUP_OUTPUT 2>&1 &

echo $NOHUP_OUTPUT
