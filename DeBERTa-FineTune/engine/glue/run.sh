#!/bin/bash

TASK=rte
MODEL=microsoft/deberta-base-mnli
EPOCHS=24
WARMUP_STEPS=100
LR=4e-6
PRUNE_FREQ=10
NOHUP_OUTPUT=outputs/$TASK/ep-$EPOCHS_lr-$LR_freq-$PRUNE_FREQ.log

OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=6,7
TOKENIZERS_PARALLELISM=true nohup

accelerate launch run_glue.py --task_name $TASK --model_type $MODEL --train_batch_size 8 --val_batch_size 8 --epochs $EPOCHS --warmup_steps $WARMUP_STEPS --lr $LR --max_seq_length 320 --pad_to_max_seq_length --cls_dropout 0.2 --weight_decay 0 --pruning --prune_frequency $PRUNE_FREQ >> $NOHUP_OUTPUT 2>&1 &
