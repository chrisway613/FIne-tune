import math

from copy import deepcopy
from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformers.trainer_utils import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION as _type_to_scheduler_func


def get_linear_schedule_with_pruning(optimizer, num_sparse_steps, num_training_steps, num_warmup_steps=0, last_epoch=-1, min_lr=0.):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_sparse_steps:
            return 1.
        else:
            return max(
                min_lr, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_sparse_steps))
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Note: must do deepcopy(instead copy) in case of modifying the original
TYPE_TO_SCHEDULER_FUNCTION = deepcopy(_type_to_scheduler_func)
TYPE_TO_SCHEDULER_FUNCTION.update(linear_with_pruning=get_linear_schedule_with_pruning)

ENUM_SHEDULER_TYPE = ('linear_with_pruning',)


def build_lr_scheduler(optimizer, config, step_per_epoch):
    # "linear", "linear_with_pruning", 
    # "cosine", "cosine_with_restarts",
    # "polynomial", 
    # "constant", "constant_with_warmup"
    scheduler_name = config.TRAIN.LR_SCHEDULER.TYPE

    update_steps_per_epoch = math.ceil(step_per_epoch / config.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    train_steps = (config.TRAIN.EPOCHS - config.TRAIN.START_EPOCH) * update_steps_per_epoch

    kwargs = {}
    if scheduler_name == 'cosine_with_restarts':
        kwargs['num_cycles'] = config.TRAIN.LR_SCHEDULER.NUM_CYCLES
    if scheduler_name == 'linear_with_pruning':
        kwargs['num_sparse_steps'] = config.PRUNE.SPARSE_STEPS
        kwargs['min_lr'] = config.TRAIN.MIN_LR

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

    if name not in ENUM_SHEDULER_TYPE:
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
