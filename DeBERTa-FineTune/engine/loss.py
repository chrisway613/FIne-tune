import torch.nn.functional as F


def soft_ce_loss(predicts, targets, temperature=1., reduction='mean'):
    likelihood = F.log_softmax(predicts / temperature, dim=-1)
    soft_targets = F.softmax(targets / temperature, dim=-1)
    
    loss = soft_targets * -likelihood
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise NotImplementedError(f"'reduction':{reduction} not implemented")


loss_dict = {
    'mse': F.mse_loss,
    'soft_ce': soft_ce_loss
}
