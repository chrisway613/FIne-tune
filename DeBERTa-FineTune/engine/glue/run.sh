#!/bin/bash

# Task & Model
TASK=rte
MODEL=deberta-base-mnli

# Train
EPOCHS=60
VAL_BS=8
TRAIN_BS=8
WARMUP_STEPS=400

# Optimization
LR=1.5e-5
OPTIMIZER=adamw
SCHEDULER=constant_linear
WEIGHT_DECAY=1e-2

# Data
TRAIN_FILE=/home/user/weicai/datasets/rte/aug/train_aug.csv
VAL_FILE=/home/user/weicai/datasets/rte/aug/dev.csv

# Prune
PRUNE_FREQ=400
SPARSE_STEPS=180800
PRUNE_DEPLOY_DEVICE=none

# Kd
TEACHER_INIT=true
TEACHER=/home/user/weicai/Fine-tune/DeBERTa-FineTune/engine/glue/aug_dense_84.38.pth

# Log
NOHUP_OUTPUT=outputs/$TASK/$MODEL/ep$EPOCHS-lr$LR-freq$PRUNE_FREQ-steps$SPARSE_STEPS.log

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true MPLBACKEND='Agg' \
  nohup accelerate launch run_glue.py --task_name $TASK --model_type microsoft/$MODEL \
  --train_file $TRAIN_FILE --train_batch_size $TRAIN_BS \
  --val_file $VAL_FILE --val_batch_size $VAL_BS \
  --epochs $EPOCHS --warmup_steps $WARMUP_STEPS \
  --lr $LR --optimizer $OPTIMIZER --lr_scheduler_type $SCHEDULER \
  --max_seq_length 320 --pad_to_max_seq_length \
  --cls_dropout 0.2 --weight_decay $WEIGHT_DECAY \
  --pruning --prune_frequency $PRUNE_FREQ \
  --sparse_steps $SPARSE_STEPS --prune_deploy_device $PRUNE_DEPLOY_DEVICE \
  --kd_on --teacher_path $TEACHER --teacher_init $TEACHER_INIT >> $NOHUP_OUTPUT 2>&1 &

echo $NOHUP_OUTPUT
