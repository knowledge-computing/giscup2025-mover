import os
import argparse
import csv
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets.dataset import TestDataset
from models.mobert import MoBERT
from utils.utils import *
from utils.options import parse_option
from utils.seq_eval import calc_geobleu

from utils.config.default import _C
from utils.options import _update_config_from_file
from models.model_factory import build_model


def run_step(model, batch, device, use_forward_infer=False):
    for k, v in batch.items():
        batch[k] = batch[k].to(device)
    if use_forward_infer:
        output = model.forward_infer(batch)
    else:
        output = model(batch)        
    return output


def generate_prediction(output):
    if output.get("pred_x1") is not None and output.get("pred_y1") is not None:
        output["pred_x"] = (output["pred_x1"] + output["pred_x2"]) / 2
        output["pred_y"] = (output["pred_y1"] + output["pred_y2"]) / 2

    if output.get("pred_x") is not None and output.get("pred_y") is not None:
        pred_x, pred_y = [], []
        pre_x, pre_y = -1, -1
        for step in range(output["pred_x"].shape[1]):
            if step > 0:
                output["pred_x"][0][step][pre_x] *= 0.9
                output["pred_y"][0][step][pre_x] *= 0.9
            pred_x.append(torch.argmax(output["pred_x"][0][step], dim=-1))
            pred_y.append(torch.argmax(output["pred_y"][0][step], dim=-1))
            pre_x, pre_y = pred_x[-1].item(), pred_y[-1].item()
        
    if output.get("pred_2d") is not None:
        H = W = 200
        B, L, HW = output["pred_2d"].shape
        pred_x, pred_y = [], []
        pre_x, pre_y = -1, -1
        for step in range(L):
            logits_step = output["pred_2d"][0, step]      # (40000,)
        
            if step > 0:
                pre_idx = pre_y * W + pre_x
                logits_step[pre_idx] *= 1 # 0.9
        
            idx = torch.argmax(logits_step)
            y = idx // W
            x = idx % W
        
            pred_x.append(x)
            pred_y.append(y)
            pre_x, pre_y = x.item(), y.item()
        
    pred_x = torch.stack(pred_x).unsqueeze(0)  # [1,L]
    pred_y = torch.stack(pred_y).unsqueeze(0)  # [1,L]
    return pred_x, pred_y


def inference():
    device = torch.device('cuda:0')

    test_dataset = TestDataset(config.TEST_CITIES, config, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             num_workers=config.TRAIN.NUM_WORKERS)
    
    model = build_model(config, device)
    weights = torch.load(os.path.join(config.CHECKPOINT_PATH, "model_best.pth"))
    msg = model.load_state_dict(weights)
    print(f"Load weights from {config.CHECKPOINT_PATH}: {msg}")
        
    results = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):

            output = run_step(model, data, device)            
            pred_x, pred_y = generate_prediction(output)
            pred = torch.stack([pred_x, pred_y], dim=-1)
            uid = data['uid'].expand(1, data['d_tgt'].shape[1]).to(pred_x.device)
            generated = torch.cat((uid.unsqueeze(-1),
                                   data['d_tgt'].unsqueeze(-1), 
                                   data['t_tgt'].unsqueeze(-1) - 1, 
                                   pred+1), dim=-1).cpu().tolist()
            generated = [tuple(x) for x in generated[0]]
            results.extend(generated)

    csv_file_path = os.path.join(args.result_path, f"{config.EXP_NAME}__{args.city_name}.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['uid', 'd', 't', 'x', 'y'])
        writer.writerows(results)

    # df = pd.read_csv(csv_file_path)
    # df.to_csv(csv_file_path + '.gz', index=False, compression="gzip")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default='cityA')
    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--result_path', type=str, default="_submission")
    args = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, args.config)

    os.makedirs(args.result_path, exist_ok=True)
    args.result_file = os.path.join(args.result_path, config.EXP_NAME + ".csv")

    inference()


# python validation.py --city_name cityD --pth_file './_runs/pretrain/models/model_final.pth'