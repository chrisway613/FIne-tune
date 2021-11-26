#!/bin/bash

TASK=rte
MODEL=deberta-base-mnli
EPOCHS=120
WARMUP_STEPS=100
LR=1e-5
WEIGHT_DECAY=0
PRUNE_FREQ=200
TEACHER_PATH='outputs/rte/deberta-base-mnli/11-26-17-48/best/epoch4.pth'
NOHUP_OUTPUT=outputs/$TASK/$MODEL/ep$EPOCHS-lr$LR-wdecay$WEIGHT_DECAY-pfreq$PRUNE_FREQ-kd.log

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true nohup accelerate launch run_glue.py --task_name $TASK --model_type microsoft/$MODEL --train_batch_size 16 --val_batch_size 16 --epochs $EPOCHS --warmup_steps $WARMUP_STEPS --lr $LR --max_seq_length 320 --pad_to_max_seq_length --cls_dropout 0.2 --weight_decay $WEIGHT_DECAY --pruning --prune_frequency $PRUNE_FREQ --kd_on --teacher_path $TEACHER_PATH >> $NOHUP_OUTPUT 2>&1 &
