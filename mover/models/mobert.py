import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from datasets.builtin import *
from models.encoder import MoBERTEncoder
from models.model_utils import *

######################################################
######################################################
######################################################
    
class MoBERT(nn.Module):
    def __init__(self, config):
        super(MoBERT, self).__init__()
        num_layers = config.MODEL.MOBERT.DECODER.NUM_LAYERS
        num_heads = config.MODEL.MOBERT.DECODER.NUM_HEADS
        emb_size = config.MODEL.MOBERT.DECODER.EMB_SIZE
        dropout = config.MODEL.MOBERT.DROPOUT            
        
        self.bert_encoder = MoBERTEncoder(config)
        self.input_encoder = TemporalEncoding(emb_size, dropout)
        
        layer = nn.TransformerDecoderLayer(
            d_model=emb_size, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        
        #################################################
        self.add_moe_module = config.MODEL.ADD_MOE_MODULE
        if self.add_moe_module:
            self.moe = MixtureOfExperts(8, emb_size)
            
        #################################################
        if config.MODEL.HEAD == 'ffn1d':
            self.xy_out_layer = FFN1DHead(emb_size)
        elif config.MODEL.HEAD == 'ffn2d':
            self.xy_out_layer = FFN2DHead(emb_size)
        elif config.MODEL.HEAD == 'coarse2fine':
            self.xy_out_layer = CoarseToFineHead(emb_size)
        elif config.MODEL.HEAD == 'conditional':
            self.xy_out_layer = ConditionalHead(emb_size)
        elif config.MODEL.HEAD == 'dual_conditional':
            self.xy_out_layer = DualConditionalHead(emb_size)
            
        #################################################
        self.predict_sep_token = 'binary' in config.TRAIN.LOSSES
        if self.predict_sep_token:
            self.binary_out_layer = BinaryTokenHead(emb_size)
    
    def _build_dec_emb(self, feats):
        day = feats['d_tgt']
        time = feats['t_tgt']
        dow = feats['dow_tgt']
        timedelta = feats['time_delta_tgt']
        emb = self.input_encoder(day, time, dow, timedelta)
        return emb

    def forward(self, batch_data):        
        memory = self.bert_encoder(batch_data)
        memory_key_padding_mask = (batch_data["d"] == PAD_TOKEN) | (batch_data["d"] == DAY_SEP_TOKEN)

        device = batch_data["d"].device
        dec_key_padding_mask = (batch_data["d_tgt"] == PAD_TOKEN) 
        
        tgt = self._build_dec_emb(batch_data)
        h = self.decoder(
            tgt=tgt,
            memory=memory,                               
            tgt_key_padding_mask=dec_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )         
        #################################################
        if self.add_moe_module:
            h = self.moe(h)            
        #################################################
        output = self.xy_out_layer(h)            
        if self.predict_sep_token:
            output["pred_sep"] = self.binary_out_layer(h)
        return output

    