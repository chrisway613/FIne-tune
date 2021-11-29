#!/bin/bash

TASK=rte
MODEL=deberta-base-mnli
EPOCHS=6
WARMUP_STEPS=100
LR=1e-6
WEIGHT_DECAY=0
PRUNE_FREQ=100
SPARSE_STEPS=1200
TEACHER_PATH=None
NOHUP_OUTPUT=outputs/$TASK/$MODEL/ep$EPOCHS-lr$LR-wdecay$WEIGHT_DECAY-pfreq$PRUNE_FREQ.log

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true nohup accelerate launch run_glue.py --task_name $TASK --model_type microsoft/$MODEL --train_batch_size 8 --val_batch_size 8 --epochs $EPOCHS --warmup_steps $WARMUP_STEPS --lr $LR --max_seq_length 320 --pad_to_max_seq_length --cls_dropout 0.2 --weight_decay $WEIGHT_DECAY --pruning --prune_frequency $PRUNE_FREQ --sparse_steps $SPARSE_STEPS >> $NOHUP_OUTPUT 2>&1 &
