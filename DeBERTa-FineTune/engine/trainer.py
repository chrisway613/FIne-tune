# --------------------------------------------------------
# [train & val for one epoch]
# Copyright (c) 2021 Moffett.AI
# Licensed under Moffett.AI
# Written by CW
# --------------------------------------------------------

import time
import torch
import datetime

from datasets.load import load_metric

from torch.cuda import max_memory_allocated
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.misc import get_grad_norm
from utils.dist import synchronize, all_reduce

from data.squad.process import postprocess_predictions


class Trainer:
    @staticmethod
    def train(model, dataloader, optimizer, lr_scheduler, config, logger, epoch, progress_bar, device):
        model.train()

        # i. Loop over batches
        start = batch_start = time.time()
        for i, batch in enumerate(dataloader):
            # Put the data into proper device
            batch = {k: v.to(device) for k, v in batch.items()}
            # ii. Model forward
            outputs = model(**batch)

            # iii. Compute loss & grad norm, backward, update parameters & lr
            loss = outputs.loss
            loss /= config.TRAIN.GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if config.TRAIN.CLIP_GRAD:
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())

            if not (i + 1) % config.TRAIN.GRADIENT_ACCUMULATION_STEPS or i == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            synchronize()
            batch_time = time.time() - batch_start

            # iv. Print logs when meet the frequency
            if not i % config.PRINT_FREQ:
                lr = optimizer.param_groups[0]['lr']
                memory_used = max_memory_allocated() / (1024. ** 2)
                logger.info(
                    f'Train Epoch[{epoch}/{config.TRAIN.EPOCHS}] Step[{i}/{len(dataloader)}]\t'
                    f'lr: {lr:.10f}\t'
                    f'batch time: {batch_time:.4f}s\t'
                    f'loss: {loss.item():.4f}\t'
                    f'grad norm: {grad_norm:.4f}\t'
                    f'memory used: {memory_used:.0f}MB\n'
                )

            batch_start = time.time()
            progress_bar.update()

            # TODO: remove this debugging intent
            break

        # v. Print epoch time
        epoch_time = time.time() - start
        logger.info(f"=> Epoch{epoch} training takes time: {datetime.timedelta(seconds=epoch_time)}\n")

        # Release gpu resources(but not be available to Pytorch)
        torch.cuda.empty_cache()
        
    @staticmethod
    def val(model, dataloader, val_data, val_features, metric_computor, config, logger, epoch, device):
        model.eval()
        
        with torch.no_grad():
            start = batch_start = time.time()
            start_logits, end_logits = [], []

            for i, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                loss = outputs.loss
                start_logits.append(outputs.start_logits)
                end_logits.append(outputs.end_logits)

                batch_time = time.time() - batch_start

                if not i % config.PRINT_FREQ:
                    memory_used = max_memory_allocated() / (1024. ** 2)
                    logger.info(
                        f'Val Epoch[{epoch}/{config.TRAIN.EPOCHS}] Step[{i}/{len(dataloader)}]\t'
                        f'batch time: {batch_time:.4f}s\t'
                        f'loss: {loss.item():.4f}\t'
                        f'memory used: {memory_used:.0f}MB\n'
                    )

                batch_start = time.time()
        
            epoch_time = time.time() - start
            logger.info(f"=> Epoch{epoch} validation takes time: {datetime.timedelta(seconds=epoch_time)}\t")

        start_logits = torch.cat(start_logits)
        end_logits = torch.cat(end_logits)
        raw_predictions = [start_logits, end_logits]

        # Postprocess in order to compute metric
        predictions = postprocess_predictions(val_data, val_features, raw_predictions, config)
        references = [{'id': example['id'], 'answers': example['answers']} for example in val_data]
        
        # Comput F1 & Exact Match
        val_results = metric_computor.compute(predictions=predictions, references=references)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Reduce results across all gpus 
            val_results_ts = {k: torch.tensor(v).cuda() for k, v in val_results.items()}
            reduced_results = all_reduce(val_results_ts)
            val_results = reduced_results

        # Release gpu resources(but not be available to Pytorch)
        torch.cuda.empty_cache()

        f1, em = val_results['f1'], val_results['exact_match']
        logger.info(f"F1: {f1:.2f} EM: {em:.2f}\n")

        return f1, em
