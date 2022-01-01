import math
import torch

from typing import Callable, Iterable

from torch.nn.parameter import Parameter
from torch.distributions import Bernoulli
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer, Adam, AdamW


def set_weight_decay(model, skip=()):
    decay, no_decay = [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(nd in name for nd in skip):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [{'params': decay}, {'params': no_decay, 'weight_decay': 0.}]


def build_optimizer(model, config):
    no_decay = model.no_decay() if hasattr(model, 'no_decay') else config.MODEL.NO_DECAY_KEYWORDS
    # Parameters distinguish decay & no-decay
    params = set_weight_decay(model, skip=no_decay)

    optimizer, opt_name = None, config.TRAIN.OPTIMIZER.NAME
    if opt_name.lower() == 'adam':
        optimizer = Adam(params, lr=config.TRAIN.LR, betas=config.TRAIN.OPTIMIZER.BETAS,
                         eps=config.TRAIN.OPTIMIZER.EPS, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_name.lower() == 'adamw':
        optimizer = AdamW(params, lr=config.TRAIN.LR, betas=config.TRAIN.OPTIMIZER.BETAS,
                          eps=config.TRAIN.OPTIMIZER.EPS, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_name.lower() == 'child_tuning_adamw':
        optimizer = ChildTuningAdamW(
            params, lr=config.TRAIN.LR, betas=config.TRAIN.OPTIMIZER.BETAS,
            eps=config.TRAIN.OPTIMIZER.EPS, weight_decay=config.TRAIN.WEIGHT_DECAY,
            reserve_p=config.TRAIN.OPTIMIZER.CHILD_TUNING_ADAMW_RESERVE_P, 
            mode=config.TRAIN.OPTIMIZER.CHILD_TUNING_ADAMW_MODE
        )
    else:
        raise NotImplementedError(f"=> Current only support 'Adam', 'AdamW'\n")

    return optimizer


class ChildTuningAdamW(Optimizer):
    def __init__(
        self, params: Iterable[Parameter], lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-6,
        weight_decay: float = 0., correct_bias: bool = True, reserve_p: float = 1., mode=None
    ):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay coefficient (default: 1e-2)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False)
            reserve_p (float, optional): map value from probability to binary mask (default: 1.0)
            mode (str, optional): 'F' for Task free, 'D' for Task driven, None for naive AdamW (default: None)

        .. _Decoupled Weight Decay Regularization:
            https://arxiv.org/abs/1711.05101
        .. _On the Convergence of Adam and Beyond:
            https://openreview.net/forum?id=ryQu7f-RZ
        """

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}, should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}, should be in [0.0, 1.0]")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}, should be in [0.0, 1.0]")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}, should be >= 0.0")
        if not 0.0 < reserve_p <= 1.0:
            raise ValueError(f"Invalid reserve p value: {reserve_p}, should be in (0.0, 1]")
        
        assert mode in (None, 'F', 'D'), f"Invalid mode value: {mode}, should be in (None, 'D', 'F')"
        if mode is None:
            import logging
            logging.warning("Please note that set mode = None equals to naive AdamW.")

        default = dict(
            lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, correct_bias=correct_bias
        )
        super().__init__(params, default)

        self.mode = mode
        self.grad_mask = None
        self.reserve_p = reserve_p
    
    def cal_fisher(self, model, dataloader, max_grad_norm=1., norm_type=2):
        """Calculate Fisher Information for different parameters"""

        # Original train mode
        train_mode = model.training
        model.train()

        # TODO: make this more general
        target_params = list(filter(lambda name_param: 'layer' in name_param[0], model.named_parameters()))

        grad_mask = dict()
        for _, params in target_params:
            grad_mask[params] = params.new_zeros(params.size())
        
        N = len(dataloader)
        for batch in dataloader:
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()

            for _, params in target_params:
                clip_grad_norm_(params, max_grad_norm, norm_type=norm_type)
                # Mean grad norm overall batches
                grad_mask[params] += (params.grad ** norm_type) / N

            # Zero grad before next batch
            model.zero_grad()
        # Recover back to the original train mode
        model.train(mode=train_mode)

        del target_params

        # Numpy
        # import numpy as np

        # r = None
        # for k, v in grad_mask.items():
        #     v = v.view(-1).cpu().numpy()
        #     if r is None:
        #         r = v
        #     else:
        #         r = np.append(r, v)

        # polar = np.percentile(r, (1 - self.reserve_p) * 100)
        # for k in grad_mask:
        #     grad_mask[k] = grad_mask[k] >= polar

        # print('Polar => {}'.format(polar))
        
        all_grad_norm = torch.cat([grad_norm.flatten() for grad_norm in grad_mask.values()])
        # Top-k norm
        k = int((1 - self.reserve_p) * all_grad_norm.size())
        polar = torch.kthvalue(all_grad_norm, k)
        for param in grad_mask:
            # Binary mask
            grad_mask[param] = (grad_mask[param] >= polar).float()
        
        return grad_mask
    
    def set_grad_mask(self, grad=None, model=None, dataloader=None, max_grad_norm=1., norm_type=2):
        if self.mode == 'F':
            assert grad is not None, f"grad value cannot be None in task free mode!"
            mask_dist = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
            grad_mask = mask_dist.sample() / self.reserve_p
        else:
            # ChildTuning-D
            assert model is not None and dataloader is not None, f"model and dataloader must be set in task driven mode!"
            grad_mask = self.cal_fisher(model=model, dataloader=dataloader, 
                                        max_grad_norm=max_grad_norm, norm_type=norm_type)
            self.grad_mask = grad_mask
        
        return grad_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step, inherit AdamW but hack gradient mask.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that re-evaluates the model and returns the loss.
        """
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================         
                if self.mode is not None:
                    if self.mode == 'ChildTuning-D':
                        # Gradient mask must be set
                        assert self.grad_mask is not None
                        if p in self.grad_mask:
                            grad *= self.grad_mask[p]
                    else:
                        # ChildTuning-F
                        grad_mask = self.set_grad_mask(grad=grad)
                        grad *= grad_mask
                # =================== HACK END =========================

                state = self.state[p]
                # State initialization
                if not len(state):
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                # This is the weight decay way of AdamW
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
