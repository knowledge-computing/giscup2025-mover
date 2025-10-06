import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertConfig

from models.model_utils import *
from datasets.builtin import *

######################################################
######################################################
######################################################

class MoBERTEncoder(nn.Module):
    def __init__(self, config):
        super(MoBERTEncoder, self).__init__()

        num_layers = config.MODEL.MOBERT.ENCODER.NUM_LAYERS
        num_heads = config.MODEL.MOBERT.ENCODER.NUM_HEADS
        emb_size = config.MODEL.MOBERT.ENCODER.EMB_SIZE
        max_seq_len = config.MODEL.MOBERT.ENCODER.MAX_SEQ_LEN
        dropout = config.MODEL.MOBERT.DROPOUT
        self.add_loc_profile = config.DATA.ADD_LOC_PROFILE

        self.input_encoder = InputEncoding(emb_size, dropout, 
                                           add_trip_seq=True, 
                                           add_loc_profile=self.add_loc_profile)
        self.bert_config = BertConfig(
            vocab_size=1,
            max_position_embeddings=max_seq_len ,
            num_attention_heads=num_heads,
            hidden_size=emb_size,
            intermediate_size=emb_size * 4,
            num_hidden_layers=num_layers)

        self.bert = BertModel(self.bert_config)

    def _build_enc_emb(self, feats):
        day = feats['d']
        time = feats['t']
        dow = feats['dow']
        timedelta = feats['time_delta']
        location_x = feats['input_x']
        location_y = feats['input_y']
        trip_seq = feats['trip_seq']
        if self.add_loc_profile:
            loc_profile = feats['loc_profile']
            emb = self.input_encoder(day, time, dow, timedelta, location_x, location_y, loc_profile, trip_seq)
        else:
            emb = self.input_encoder(day, time, dow, timedelta, location_x, location_y, trip_seq)
        return emb
        
    def forward(self, batch_data):
        emb = self._build_enc_emb(batch_data)
        attention_mask = (batch_data['d'] != PAD_TOKEN).long()
        outputs = self.bert(inputs_embeds=emb,
                            attention_mask=attention_mask)
        return outputs.last_hidden_state


class MoBERTMLM(nn.Module):
    def __init__(self, config):
        super(MoBERTMLM, self).__init__()
        emb_size = config.MODEL.MOBERT.ENCODER.EMB_SIZE    
        self.encoder = MoBERTEncoder(config)
        self.add_city_emb = config.DATA.ADD_CITY_EMB
        if self.add_city_emb:
            self.city_embedding = CityEmbedding(config.DATA.CITY_EMB_SIZE)  
            self.out_layer = FFN1DHead(emb_size + config.DATA.CITY_EMB_SIZE)
        else:
            self.out_layer = FFN1DHead(emb_size)

    def forward(self, batch_data):
        h = self.encoder(batch_data)
        if self.add_city_emb:
            city_emb = self.city_embedding(batch_data['city'])
            h = torch.cat([h, city_emb], dim=-1)
                
        return self.out_layer(h)





