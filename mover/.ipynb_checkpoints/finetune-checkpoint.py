import os
import sys
import shutil
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_scheduler
import wandb

from datasets.dataset import FinetuneDataset
from models.model_factory import build_model
from models.losses import get_loss

from utils.utils import *
from utils.options import parse_option


def run_step(model, batch, device, use_forward_infer=False):
    for k, v in batch.items():
        batch[k] = batch[k].to(device)
    if use_forward_infer:
        output = model.forward_infer(batch)
    else:
        output = model(batch)
    return output


def train():
    device = torch.device('cuda:0')

    if config.USE_WANDB:
        wandb.init(project="HUMOB", name=config.EXP_NAME)
        wandb.run.name = config.EXP_NAME

    train_dataset = FinetuneDataset(config.TRAIN_CITIES, config, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, 
                              collate_fn=finetune_collate_fn, num_workers=config.TRAIN.NUM_WORKERS)
    val_dataset = FinetuneDataset(config.TEST_CITIES, config, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            collate_fn=finetune_collate_fn, num_workers=config.TRAIN.NUM_WORKERS)
    print(f"Number of train samples: {len(train_dataset)}; Number of val samples: {len(val_dataset)}")
    print(f"Number of train batches: {len(train_loader)}; Number of val batches: {len(val_loader)}")

    model = build_model(config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, n_iter_per_epoch=len(train_loader))

    #######################
    ##### Train Model #####
    #######################
    best_val_loss = np.inf
    epochs_no_improve = 0  
    for epoch_id in range(config.TRAIN.EPOCHS + 1):
        print(f"... Epoch={epoch_id} Starts ...")
        
        model.train()
        epoch_loss_dict = defaultdict(int)
        for batch_id, batch in enumerate(tqdm(train_loader)):
            output = run_step(model, batch, device)
            loss_dict, loss = get_loss(output, batch, losses=config.TRAIN.LOSSES)
            for k, v in loss_dict.items():
                epoch_loss_dict[k] += v

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step_update(epoch_id * len(train_loader) + batch_id)
            
            step = epoch_id * len(train_loader) + batch_id
            for k, v in loss_dict.items():
                wandb_logging({k: v, "step": step}, config);

            if step % 200 == 0:
                torch.cuda.empty_cache() 
                torch.cuda.reset_peak_memory_stats()
            
        for k, v in epoch_loss_dict.items():
            wandb_logging({k: v / len(train_loader), "epoch": epoch_id}, config)
            print(f"Epoch-{epoch_id} | Train: {k}, {v / len(train_loader)}")

        
        ######################
        ##### Eval Model #####
        ######################
        if config.EVAL_EVERY_EPOCH > 0 and epoch_id % config.EVAL_EVERY_EPOCH == 0:
            model.eval()
            with torch.no_grad():
                epoch_loss_dict = defaultdict(int)
                for batch_id, batch in enumerate(tqdm(val_loader)):
                    if 'seq2seq' in config.MODEL_NAME:
                        model.p_corrupt = 0.2
                        output = run_step(model, batch, device, use_forward_infer=True)
                    else:
                        output = run_step(model, batch, device)

                    loss_dict, _ = get_loss(output, batch, losses=config.TRAIN.LOSSES)
                    for k, v in loss_dict.items():
                        epoch_loss_dict[k] += v
                    
                for k, v in epoch_loss_dict.items():
                    wandb_logging({"val_" + k: v / len(val_loader), "epoch": epoch_id}, config)
                    print(f"Epoch-{epoch_id} | Val: {k}, {v / len(val_loader)}")

                epoch_val_loss = epoch_loss_dict[config.TRAIN.BASE_LOSS] / len(val_loader)
                if epoch_val_loss < best_val_loss:
                    print(f"Accuracy improved in val loss: {abs(epoch_val_loss-best_val_loss)}.")
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                    model_save_path = os.path.join(config.CHECKPOINT_PATH, f'model_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Best model saved at epoch {epoch_id}")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement in val loss for {epochs_no_improve} epochs.")
                
        if epoch_id > 0 and epoch_id % config.SAVE_EVERY_EPOCH == 0:
            model_save_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch{epoch_id}.pth')
            torch.save(model.state_dict(), model_save_path)

        if epochs_no_improve >= config.TRAIN.PATIENCE:
            print(f"No improvement in val loss for {config.TRAIN.PATIENCE} epochs. Early stopping.")
            sys.exit(0)


if __name__ == '__main__':
    config = parse_option()    
    set_random_seed(config.SEED)
    train()
