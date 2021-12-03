#!/bin/bash

TASK=rte
MODEL=deberta-base-mnli
EPOCHS=24
WARMUP_STEPS=20
LR=7e-6
WEIGHT_DECAY=0
PRUNE_FREQ=25
SPARSE_STEPS=500
TEACHER_PATH=''
NOHUP_OUTPUT=outputs/$TASK/$MODEL/ep$EPOCHS-lr$LR-wdecay$WEIGHT_DECAY-pfreq$PRUNE_FREQ.log

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true nohup accelerate launch run_glue.py --task_name $TASK --model_type microsoft/$MODEL --train_batch_size 8 --val_batch_size 8 --epochs $EPOCHS --warmup_steps $WARMUP_STEPS --lr $LR --max_seq_length 320 --pad_to_max_seq_length --cls_dropout 0.2 --weight_decay $WEIGHT_DECAY --pruning --prune_frequency $PRUNE_FREQ --sparse_steps $SPARSE_STEPS >> $NOHUP_OUTPUT 2>&1 &

tail -f $NOHUP_OUTPUT
