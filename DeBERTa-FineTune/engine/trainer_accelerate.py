import time
import torch
import datetime

from torch.cuda import max_memory_allocated
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.misc import get_grad_norm


class Trainer:
    @classmethod
    def train(cls, accelerator, model, dataloader, optimizer, lr_scheduler, 
              config, logger, epoch, progress_bar, pruner=None, teacher=None, 
              kd_cls_loss=None, kd_reg_loss=None):
        model.train()

        start = batch_start = time.time()
        for step, batch in enumerate(dataloader):
            # Forward
            outputs = model(**batch)

            # Kd
            if teacher is not None:
                assert kd_cls_loss is not None and kd_reg_loss is not None, \
                    "'kd_cls_loss' & 'kd_reg_loss' must be set"

                # Pay attention to set 'no_gread' for teacher forwarding
                with torch.no_grad():
                    teacher_outputs = teacher(**batch)
                    teacher_hidden_states, teacher_attns, teacher_logits = \
                        teacher_outputs.hidden_states, teacher_outputs.attentions, \
                            teacher_outputs.logits

                hidden_states, attns, logits = outputs.hidden_states, \
                    outputs.attentions, outputs.logits

                # TODO: gather these losses?
                # Logits kd loss(ce)
                loss_raw = kd_cls_loss(logits, teacher_logits)
                # Hidden states kd loss(mse)
                for layer_hidden_state, teacher_layer_hidden_state in \
                    zip(hidden_states[config.KD.TRAIN.BEGIN_LAYER:], teacher_hidden_states[config.KD.TRAIN.BEGIN_LAYER:]):
                    loss_raw += kd_reg_loss(layer_hidden_state, teacher_layer_hidden_state)
                # Attentions kd loss(mse)
                for layer_attn, teacher_layer_attn in \
                    zip(attns[config.TRAIN.KD.BEGIN_LAYER:], teacher_attns[config.TRAIN.KD.BEGIN_LAYER:]):
                    loss_raw += kd_reg_loss(layer_attn, teacher_layer_attn)
            else:
                loss_raw = outputs.loss
            
            # Gradient accumulation
            loss = loss_raw / config.TRAIN.GRADIENT_ACCUMULATION_STEPS
            # Backward
            accelerator.backward(loss)
            # Clip gradient(optional)
            if config.TRAIN.CLIP_GRAD:
                grad_norm = clip_grad_norm_(
                    accelerator.unwrap_model(model).parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(accelerator.unwrap_model(model).parameters())

            # Update parameters, lr, zero gradients, pruning(optional)
            if not (step + 1) % config.TRAIN.GRADIENT_ACCUMULATION_STEPS or step == len(dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if pruner is not None:
                    # This is for old pruner
                    # pruner.prune()
                    cur_sparsity = pruner.prune()
                    if cur_sparsity is not None:
                        logger.info(f"=> current sparsity: {cur_sparsity}")

                    if pruner._update_mask_conditions():
                        # This is for old pruner
                        # layer_sparse_rate, total_sparse_rate = pruner.prune_sparsity()
                        layer_sparse_rate, total_sparse_rate = pruner.sparsity()
                        logger.info(f'\nweight sparsity: {total_sparse_rate}\n'
                                    f'layer weight sparsity:\n{layer_sparse_rate}\n')

                progress_bar.update(1)
            
            accelerator.wait_for_everyone()
            batch_time = time.time() - batch_start

            if not step % config.PRINT_FREQ:
                lr = optimizer.param_groups[0]['lr']
                memory_used = max_memory_allocated() / (1024. ** 2)

                logger.info(
                    f'Train Epoch[{epoch}/{config.TRAIN.EPOCHS}] Step[{step}/{len(dataloader)}]\t'
                    f'lr: {lr:.10f}\t'
                    f'batch time: {batch_time:.4f}s\t'
                    f'loss raw: {loss_raw.item():.4f}\t'
                    f'loss(w gradient accumulate): {loss.item():.4f}\t'
                    f'grad norm: {grad_norm:.4f}\t'
                    f'memory used: {memory_used:.0f}MB\n'
                )
            
            batch_time = time.time()
        
        epoch_time = time.time() - start
        logger.info(f"=> Epoch{epoch} training takes time: {datetime.timedelta(seconds=epoch_time)}\n")

    @torch.no_grad
    @classmethod
    def val(cls, accelerator, model, dataloader, config, logger, 
            epoch, metric_computor, is_regression):
        model.eval()
        
        start = batch_start = time.time()
        for step, batch in enumerate(dataloader):
            # loss, logits, hidden_states, attentions
            outputs = model(**batch)
            loss = outputs.loss

            predictions = outputs.logits.argmax(dim=-1) \
                if not is_regression else outputs.logits.squeeze()
            metric_computor.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

            batch_time = time.time() - batch_start

            if not step % config.PRINT_FREQ:
                memory_used = max_memory_allocated() / (1024. ** 2)
                logger.info(
                    f'Val Epoch[{epoch}/{config.TRAIN.EPOCHS}] Step[{step}/{len(dataloader)}]\t'
                    f'batch time: {batch_time:.4f}s\t'
                    f'loss: {loss.item():.4f}\t'
                    f'memory used: {memory_used:.0f}MB\n'
                )

            batch_start = time.time()
        
        epoch_time = time.time() - start
        val_results = metric_computor.compute()
        logger.info(f"=> Epoch{epoch} metric: {val_results} \
            validation takes time: {datetime.timedelta(seconds=epoch_time)}\t")

        return val_results
