import os
import torch

from typing import List


def auto_resume_helper(checkpoint_dir):
    all_checkpoints = [ckp for ckp in os.listdir(checkpoint_dir) 
                       if (ckp.endswith('.pth') or ckp.endswith(".bin"))]
    if not all_checkpoints:
        return None
    
    return max([os.path.join(checkpoint_dir, ckp) for ckp in all_checkpoints], key=os.path.getmtime)


def load_checkpoint(model, optimizer, lr_scheduler, config, logger):
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if 'epoch' in checkpoint:
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> Resume from epoch{checkpoint['epoch']}")
    
    f1 = em = 0.
    if 'f1' in checkpoint:
        f1 = checkpoint['max_accuracy']
    if 'em' in checkpoint:
        em = checkpoint['em']

    del checkpoint
    torch.cuda.empty_cache()

    return f1, em


def save_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler, epoch, config, f1, em):
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_dir = os.path.join(checkpoint_dir, f'epoch{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    # Done by Transformers API, this will save 'config.json' & 'pytorch_model.bin' below directory
    model.save_pretrained(epoch_dir)

    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config,
                  'f1': f1, 'em': em}

    checkpoint = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
    torch.save(save_state, checkpoint)
    
    return checkpoint


def get_grad_norm(parameters: List[torch.Tensor], norm_type: float = 2.) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [param for param in parameters if param.grad is not None]

    total_norm = 0.
    for param in parameters:
        norm = param.grad.data.norm(p=norm_type).item()
        total_norm += norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm
