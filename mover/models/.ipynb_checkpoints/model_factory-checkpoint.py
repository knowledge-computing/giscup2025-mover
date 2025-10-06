import os
import torch


def build_model(config, device):
    
    if 'seq2seq' in config.MODEL_NAME:
        from models.mobert_seq2seq import MoBERT
        print("... Use MoBERT Seq2Seq model ...")
        model = MoBERT(config).to(device)
    else:
        from models.mobert import MoBERT
        print("... Use MoBERT model ...")
        model = MoBERT(config).to(device)

    if os.path.exists(config.MODEL.MOBERT.ENCODER.PRETRAIN_WEIGHT):
        weights = torch.load(config.MODEL.MOBERT.ENCODER.PRETRAIN_WEIGHT)
        weights = {k[len('encoder.'):]: v for k, v in weights.items() if k.startswith('encoder.')}
        msg = model.bert_encoder.load_state_dict(weights)
        print(f"Load pretrained weights from {config.MODEL.MOBERT.ENCODER.PRETRAIN_WEIGHT}: {msg}")

    if os.path.exists(config.MODEL.PRETRAIN_WEIGHT):
        weights = torch.load(config.MODEL.PRETRAIN_WEIGHT)
        # weights = {k: v for k, v in weights.items()}
        msg = model.load_state_dict(weights, strict=True)
        print(f"Load pretrained weights from {config.MODEL.PRETRAIN_WEIGHT}: {msg}")
        
    return model