from transformers import get_scheduler

from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

from config import WARMUP_STEPS


def build_lr_scheduler(optimizer, config, step_per_epoch):
    num_steps = config.TRAIN.EPOCHS * step_per_epoch
    decay_steps = config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * step_per_epoch

    if config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        return get_scheduler(
            config.TRAIN.LR_SCHEDULER.NAME,
            optimizer=optimizer,
            num_warmup_steps=config.TRAIN.WARMUP_STEPS,
            num_training_steps=num_steps
        )

    if config.TRAIN.LR_SCHEDULER.NAME == 'consine':
        return CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_t=config.TRAIN.WARMUP_STEPS,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            cycle_limit=1,
            t_in_epochs=False
        )

    if config.TRAIN.LR_SCHEDULER.NAME == 'step':
        return StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_t=WARMUP_STEPS,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            t_in_epochs=False
        )
