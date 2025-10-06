import os
import argparse
import time
import yaml
import shutil

from utils.config.default import _C

# -----------------------------------------------------------------------------
# Load Configuration
# -----------------------------------------------------------------------------
def _update_config_from_file(config, cfg_file):
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def update_config(config, args):
    _update_config_from_file(config, args.config)

    # if args.opts:
    #     config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    config.USE_WANDB = args.use_wandb
    if args.batch_size > 0:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if len(args.exp_name) > 0:
        config.EXP_NAME = args.exp_name
    if args.tar_cluster > -1:
        config.DATA.TAR_CLUSTER = args.tar_cluster        
    if len(args.pretrain_weight) > 0:
        config.MODEL.PRETRAIN_WEIGHT = args.pretrain_weight     
        
    # output folder
    config.EXP_DIR = os.path.join(config.OUTPUT, config.EXP_NAME)
    config.LOG_PATH = os.path.join(config.EXP_DIR, 'log')
    config.CHECKPOINT_PATH = os.path.join(config.EXP_DIR, 'checkpoint')
    config.SAVE_CONFIG_FILE = os.path.join(config.EXP_DIR, 'config.yaml')  
    # config.freeze()

    
def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--tar_cluster', type=int, default=-1)
    parser.add_argument('--pretrain_weight', type=str, default="")
    
    args = parser.parse_args()
    config = get_config(args)

    if os.path.exists(config.EXP_DIR):
        shutil.rmtree(config.EXP_DIR)
    os.makedirs(config.EXP_DIR)
    os.makedirs(config.CHECKPOINT_PATH)
    with open(config.SAVE_CONFIG_FILE, 'w') as f:
        f.write(config.dump())
    print(f"Configuration saved to {config.SAVE_CONFIG_FILE}")
    return config

