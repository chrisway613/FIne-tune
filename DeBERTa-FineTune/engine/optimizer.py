from torch.optim import Adam, AdamW


def set_weight_decay(model, skip=()):
    decay, no_decay = [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(nd in name for nd in skip):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]


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
    else:
        raise NotImplementedError(f"=> Current only support 'Adam', 'AdamW'\n")

    return optimizer
