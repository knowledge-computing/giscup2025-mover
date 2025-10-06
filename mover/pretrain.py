import os
import shutil
import argparse
import logging
import random
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_scheduler
import wandb

from datasets.dataset import PretrainDataset
from models.encoder import MoBERTMLM
from utils.options import parse_option
from utils.utils import *


def run_step(model, batch, device):
    for k, v in batch.items():
        batch[k] = batch[k].to(device)
    output = model(batch)
    return output


def get_loss(output, batch, loss_fn):
    pred_x, pred_y = output['pred_x'], output['pred_y']
    keep = (batch['input_x'] == 201)
    loss = loss_fn(pred_x[keep], batch['label_x'][keep])
    loss += loss_fn(pred_y[keep], batch['label_y'][keep])
    return loss


def train():
    device = torch.device(f'cuda:0')

    if config.USE_WANDB:
        wandb.init(project="HUMOB", name=config.EXP_NAME)
        wandb.run.name = config.EXP_NAME

    train_dataset = PretrainDataset(config.TRAIN_CITIES, config, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, 
                              collate_fn=pretrain_collate_fn, num_workers=config.TRAIN.NUM_WORKERS)
    val_dataset = PretrainDataset(config.TEST_CITIES, config, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            collate_fn=pretrain_collate_fn, num_workers=config.TRAIN.NUM_WORKERS)
    print(f"Number of train samples: {len(train_dataset)}; Number of val samples: {len(val_dataset)}")
    print(f"Number of train batches: {len(train_loader)}; Number of val batches: {len(val_loader)}")

    model = MoBERTMLM(config).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, n_iter_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = np.inf
    for epoch_id in range(config.TRAIN.EPOCHS + 1):
        print(f"Epoch={epoch_id} Starts ...")
        
        model.train()
        batch_base_loss = 0.
        for batch_id, batch in enumerate(tqdm(train_loader)):
            loss = 0
            output = run_step(model, batch, device)
            base_loss = get_loss(output, batch, criterion)
            batch_base_loss += base_loss.detach().item()
            loss += base_loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step_update(epoch_id * len(train_loader) + batch_id)
            
            step = epoch_id * len(train_loader) + batch_id
            wandb_logging({"loss": base_loss.detach().item(), "step": step}, config)
            
        wandb_logging({"epoch_train_loss": batch_base_loss / len(train_loader), "epoch": epoch_id}, config)

        if config.EVAL_EVERY_EPOCH > 0 and epoch_id % config.EVAL_EVERY_EPOCH == 0:
            model.eval()
            with torch.no_grad():
                batch_base_loss = 0.
                for batch_id, batch in enumerate(tqdm(val_loader)):
                    output = run_step(model, batch, device)
                    base_loss = get_loss(output, batch, criterion)
                    batch_base_loss += base_loss.detach().item()
                    
                wandb_logging({"epoch_val_loss": batch_base_loss / len(val_loader), "epoch": epoch_id}, config)
                    
                if batch_base_loss / len(val_loader) < best_val_loss:
                    best_val_loss = batch_base_loss / len(val_loader)
                    model_save_path = os.path.join(config.CHECKPOINT_PATH, f'model_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Best model saved at epoch {epoch_id}")

        if epoch_id > 0 and epoch_id % config.SAVE_EVERY_EPOCH == 0:
            model_save_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch{epoch_id}.pth')
            torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    config = parse_option()    
    set_random_seed(config.SEED)
    train()

    