import math

from typing import Optional, Union

from torch.optim import Optimizer

from transformers.trainer_utils import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION


def build_lr_scheduler(optimizer, config, step_per_epoch):
    # "linear", "cosine", "cosine_with_restarts",
    # "polynomial", "constant","constant_with_warmup"
    scheduler_name = config.TRAIN.LR_SCHEDULER.TYPE

    update_steps_per_epoch = math.ceil(step_per_epoch / config.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    train_steps = (config.TRAIN.EPOCHS - config.TRAIN.START_EPOCH) * update_steps_per_epoch

    kwargs = {}
    if scheduler_name == 'cosine_with_restarts':
        kwargs['num_cycles'] = config.TRAIN.LR_SCHEDULER.NUM_CYCLES

    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=config.TRAIN.WARMUP_STEPS,
        num_training_steps=train_steps,
        **kwargs
    ), train_steps


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (:obj:`str` or `:obj:`SchedulerType`):
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)
