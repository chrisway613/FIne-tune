import math

from transformers import get_scheduler


def build_lr_scheduler(optimizer, config, step_per_epoch):
    update_steps_per_epoch = math.ceil(step_per_epoch / config.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    train_steps = (config.TRAIN.EPOCHS - config.TRAIN.START_EPOCH) * update_steps_per_epoch

    return get_scheduler(
        config.TRAIN.LR_SCHEDULER.TYPE,
        optimizer=optimizer,
        num_warmup_steps=config.TRAIN.WARMUP_STEPS,
        num_training_steps=train_steps
    ), train_steps
