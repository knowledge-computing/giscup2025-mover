import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from timm.scheduler.cosine_lr import CosineLRScheduler
import wandb


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wandb_logging(content, config):
    if config.USE_WANDB:
        wandb.log(content)
        
def pretrain_collate_fn(batch):
    d = [item['d'] for item in batch]
    t = [item['t'] for item in batch]
    dow = [item['dow'] for item in batch]
    input_x = [item['input_x'] for item in batch]
    input_y = [item['input_y'] for item in batch]
    time_delta = [item['time_delta'] for item in batch]
    city = [item['city'] for item in batch]
    label_x = [item['label_x'] for item in batch]
    label_y = [item['label_y'] for item in batch]
    trip_seq = [item['trip_seq'] for item in batch]
    loc_profile = [item['loc_profile'] for item in batch]
    
    return {
        'd': pad_sequence(d, batch_first=True, padding_value=0),
        't': pad_sequence(t, batch_first=True, padding_value=0),
        'dow': pad_sequence(dow, batch_first=True, padding_value=0),
        'input_x': pad_sequence(input_x, batch_first=True, padding_value=0),
        'input_y': pad_sequence(input_y, batch_first=True, padding_value=0),
        'time_delta': pad_sequence(time_delta, batch_first=True, padding_value=0),
        'city': pad_sequence(city, batch_first=True, padding_value=0),
        'label_x': pad_sequence(label_x, batch_first=True, padding_value=0),
        'label_y': pad_sequence(label_y, batch_first=True, padding_value=0),
        'trip_seq': pad_sequence(trip_seq, batch_first=True, padding_value=0),
        'loc_profile': pad_sequence(loc_profile, batch_first=True, padding_value=0)
    }


def finetune_collate_fn(batch):
    d = [item['d'] for item in batch]
    t = [item['t'] for item in batch]
    dow = [item['dow'] for item in batch]
    time_delta = [item['time_delta'] for item in batch]
    trip_seq = [item['trip_seq'] for item in batch]
    input_x = [item['input_x'] for item in batch]
    input_y = [item['input_y'] for item in batch]
    d_tgt = [item['d_tgt'] for item in batch]
    t_tgt = [item['t_tgt'] for item in batch]
    dow_tgt = [item['dow_tgt'] for item in batch]
    time_delta_tgt = [item['time_delta_tgt'] for item in batch]
    input_x_tgt = [item['input_x_tgt'] for item in batch]
    input_y_tgt = [item['input_y_tgt'] for item in batch]
    label_x_tgt = [item['label_x_tgt'] for item in batch]
    label_y_tgt = [item['label_y_tgt'] for item in batch]
    label_sep_token = [item['label_sep_token'] for item in batch]
    loc_profile = [item['loc_profile'] for item in batch]

    return {
        'd': pad_sequence(d, batch_first=True, padding_value=0),
        't': pad_sequence(t, batch_first=True, padding_value=0),
        'dow': pad_sequence(dow, batch_first=True, padding_value=0),
        'time_delta': pad_sequence(time_delta, batch_first=True, padding_value=0),
        'trip_seq': pad_sequence(trip_seq, batch_first=True, padding_value=0),
        'input_x': pad_sequence(input_x, batch_first=True, padding_value=0),
        'input_y': pad_sequence(input_y, batch_first=True, padding_value=0),
        'd_tgt': pad_sequence(d_tgt, batch_first=True, padding_value=0),
        't_tgt': pad_sequence(t_tgt, batch_first=True, padding_value=0),
        'dow_tgt': pad_sequence(dow_tgt, batch_first=True, padding_value=0),
        'time_delta_tgt': pad_sequence(time_delta_tgt, batch_first=True, padding_value=0),
        'input_x_tgt': pad_sequence(input_x_tgt, batch_first=True, padding_value=0),
        'input_y_tgt': pad_sequence(input_y_tgt, batch_first=True, padding_value=0),
        'label_x_tgt': pad_sequence(label_x_tgt, batch_first=True, padding_value=0),
        'label_y_tgt': pad_sequence(label_y_tgt, batch_first=True, padding_value=0),
        'label_sep_token': pad_sequence(label_sep_token, batch_first=True, padding_value=0),
        'loc_profile': pad_sequence(loc_profile, batch_first=True, padding_value=0)        
    }


def build_param_groups(
    model,
    base_lr: float,
    weight_decay: float,
    encoder_prefixes=("encoder", "bert", "bert_encoder"), 
    enc_lr_scale: float = 0.1,
    no_decay_names=("bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight")
):
    """
    Returns param groups for AdamW:
      - non-encoder decay / no-decay @ base_lr
      - encoder     decay / no-decay @ base_lr * enc_lr_scale
    """

    enc_decay, enc_no_decay, dec_decay, dec_no_decay = [], [], [], []

    def is_encoder_param(name: str) -> bool:
        return any(name.startswith(pref) for pref in encoder_prefixes)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = any(name.endswith(nd) or nd in name for nd in no_decay_names)
        in_encoder = is_encoder_param(name)

        if in_encoder:
            (enc_no_decay if is_no_decay else enc_decay).append(p)
        else:
            (dec_no_decay if is_no_decay else dec_decay).append(p)

    groups = []
    if dec_decay:
        groups.append({"params": dec_decay, "weight_decay": weight_decay, "lr": base_lr})
    if dec_no_decay:
        groups.append({"params": dec_no_decay, "weight_decay": 0.0,       "lr": base_lr})
    if enc_decay:
        groups.append({"params": enc_decay, "weight_decay": weight_decay, "lr": base_lr * enc_lr_scale})
    if enc_no_decay:
        groups.append({"params": enc_no_decay, "weight_decay": 0.0,       "lr": base_lr * enc_lr_scale})

    return groups


def build_optimizer_and_scheduler(model, config, n_iter_per_epoch):

    groups = build_param_groups(model, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY)

    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        optimizer = torch.optim.AdamW(groups,
                                      eps=config.TRAIN.OPTIMIZER.EPS, 
                                      betas=config.TRAIN.OPTIMIZER.BETAS,
                                      lr=config.TRAIN.BASE_LR, 
                                      weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError
    
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
        warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False)
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler
