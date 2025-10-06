import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from datasets.builtin import *


class CyclicalEncoding(nn.Module):
    def __init__(self, emb_size, period, init_std=0.02):
        super().__init__()
        self.period = period
        self.proj = nn.Linear(2, emb_size)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_feat):  # shape (B,)
        theta = 2.0 * torch.pi * input_feat.to(torch.float32) / self.period
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        out = torch.stack([sin, cos], dim=-1)  # shape (B, L, 2)
        out = self.proj(out)
        mask = (input_feat == TIME_SEP_TOKEN) | (input_feat == PAD_TOKEN)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        return out 


class FFN1DHead(nn.Module):
    def __init__(self, emb_size):
        super(FFN1DHead, self).__init__()
        
        self.ffn1 = nn.Sequential(
            nn.Linear(emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
    def forward(self, h):
        out_x = self.ffn1(h)
        out_y = self.ffn2(h)
        return {'pred_x': out_x, 'pred_y': out_y}


class FFN2DHead(nn.Module):
    def __init__(self, emb_size):
        super(FFN2DHead, self).__init__()

        self.ffn1 = nn.Sequential(
            nn.Linear(emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
    def forward(self, h):
        out_x = self.ffn1(h)
        out_y = self.ffn2(h)
        L = out_x.unsqueeze(3) + out_y.unsqueeze(2)
        return {'pred_2d': L.view(h.size(0), h.size(1), -1)}


class ConditionalHead(nn.Module):
    def __init__(self, emb_size):
        super(ConditionalHead, self).__init__()

        self.temperature = 1.
        self.inner_emb_size = 64
        
        self.ffn_x = nn.Sequential(
            nn.Linear(emb_size, self.inner_emb_size),
            nn.ReLU(),
            nn.Linear(self.inner_emb_size, 200),
        )
        self.x_embed = nn.Embedding(200, self.inner_emb_size)

        # p(y | h, x) where x is softly represented as E[p(x)] over x_embed
        self.ffn_y = nn.Sequential(
            nn.Linear(emb_size + self.inner_emb_size, self.inner_emb_size), 
            nn.ReLU(),
            nn.Linear(self.inner_emb_size, 200)
        )

    def _act(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)
            
    def forward(self, h):
        logits_x = self.ffn_x(h)
        probs_x  = self._act(logits_x)  
        x_ctx = probs_x @ self.x_embed.weight
        logits_y = self.ffn_y(torch.cat([h, x_ctx], dim=-1))
        return {'pred_x': logits_x, 'pred_y': logits_y}


class DualConditionalHead(nn.Module):
    def __init__(self, emb_size):
        super(DualConditionalHead, self).__init__()

        self.temperature = 1.
        self.inner_emb_size = 64
        
        self.ffn_x = nn.Sequential(
            nn.Linear(emb_size, self.inner_emb_size),
            nn.ReLU(),
            nn.Linear(self.inner_emb_size, 200))
        self.x_embed = nn.Embedding(200, self.inner_emb_size)
        self.ffn_y = nn.Sequential(
            nn.Linear(emb_size + self.inner_emb_size, self.inner_emb_size), 
            nn.ReLU(),
            nn.Linear(self.inner_emb_size, 200))

        self.ffn_y1 = nn.Sequential(
            nn.Linear(emb_size, self.inner_emb_size),
            nn.ReLU(),
            nn.Linear(self.inner_emb_size, 200))
        self.y_embed = nn.Embedding(200, self.inner_emb_size)

        self.ffn_x1 = nn.Sequential(
            nn.Linear(emb_size + self.inner_emb_size, self.inner_emb_size), 
            nn.ReLU(),
            nn.Linear(self.inner_emb_size, 200))

    def _act(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)
            
    def forward(self, h):
        logits_x = self.ffn_x(h)
        probs_x  = self._act(logits_x)  
        x_ctx = probs_x @ self.x_embed.weight
        logits_y = self.ffn_y(torch.cat([h, x_ctx], dim=-1))

        logits_y1 = self.ffn_y1(h)
        probs_y1  = self._act(logits_y1)  
        y_cty = probs_y1 @ self.y_embed.weight
        logits_x1 = self.ffn_x1(torch.cat([h, y_cty], dim=-1))
        
        return {'pred_x1': logits_x, 'pred_y1': logits_y, 'pred_x2': logits_x1, 'pred_y2': logits_y1}


class CoarseToFineHead(nn.Module):
    def __init__(self, emb_size, H=200, W=200, Hc=50, Wc=50):
        super(CoarseToFineHead, self).__init__()
        self.H, self.W = H, W
        self.Hc, self.Wc = Hc, Wc
        
        self.coarse = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2), 
            nn.ReLU(),
            nn.Linear(emb_size * 2, Hc*Wc)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(H,W), mode='bilinear', align_corners=False),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, h):
        B, L, _ = h.shape
        coarse_logits = self.coarse(h).view(B * L, 1, self.Hc, self.Wc)
        fine_logits = self.refine(coarse_logits).flatten(2).view(B, L, self.H * self.W)
        return {'pred_2d': fine_logits}

        
class BinaryTokenHead(nn.Module):
    def __init__(self, d_model, hidden_mult=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, 1)  # logits per token
        )

    def forward(self, h):
        return self.net(h)


def check_ids(name, ids, num_embeddings):
    if not torch.is_floating_point(ids):
        min_id = int(ids.min().item())
        max_id = int(ids.max().item())
        assert 0 <= min_id and max_id < num_embeddings, \
            f"{name} out of range: [{min_id}, {max_id}] for vocab {num_embeddings}"


class CityEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(CityEmbedding, self).__init__()
        self.city_embedding = nn.Embedding(
            num_embeddings=CITY_SEP_TOKEN + 1,
            embedding_dim=emb_size
        )
    def forward(self, city):
        check_ids("city", city, CITY_SEP_TOKEN+1)
        embed = self.city_embedding(city)
        return embed

        
class TemporalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1):
        super(TemporalEncoding, self).__init__()
        
        self.day_embedding = nn.Embedding(DAY_SEP_TOKEN+1, emb_size // 4, padding_idx=PAD_TOKEN)
        self.time_embedding = nn.Embedding(TIME_SEP_TOKEN+1, emb_size // 4 // 2, padding_idx=PAD_TOKEN)
        self.time_cyc_embedding = CyclicalEncoding(emb_size // 4 // 2, period=48)
        self.dow_embedding = nn.Embedding(DOW_SEP_TOKEN+1, emb_size // 4 // 2, padding_idx=PAD_TOKEN)
        self.dow_cyc_embedding = CyclicalEncoding(emb_size // 4 // 2, period=7)
        self.timedelta_embedding = nn.Embedding(TIME_SEP_TOKEN+1, emb_size // 4, padding_idx=PAD_TOKEN)
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, day, time, dow, timedelta):

        check_ids("day", day, DAY_SEP_TOKEN+1)
        check_ids("time", time, TIME_SEP_TOKEN+1)
        check_ids("dow", dow, DOW_SEP_TOKEN+1)
        check_ids("timedelta", timedelta, TIME_SEP_TOKEN+1)

        day_emb = self.day_embedding(day)
        time_emb = self.time_embedding(time)
        time_cyc_emb = self.time_cyc_embedding(time)
        dow_emb = self.dow_embedding(dow)
        dow_cyc_emb = self.dow_cyc_embedding(dow)
        timedelta_emb = self.timedelta_embedding(timedelta)
        temporal_emb = torch.cat([day_emb, time_emb, time_cyc_emb, dow_emb, dow_cyc_emb, timedelta_emb], dim=-1)
        temporal_emb = self.temporal_proj(temporal_emb)
        return temporal_emb


class SpatialEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, add_loc_profile=False):
        super(SpatialEncoding, self).__init__()

        self.add_loc_profile = add_loc_profile
        if self.add_loc_profile:
            self.emb_size = emb_size // 4
            self.loc_profile_layer = nn.Linear(128, emb_size // 2)
        else:
            self.emb_size = emb_size // 2
        
        self.location_x_embedding = nn.Embedding(XY_SEP_TOKEN+1, self.emb_size, padding_idx=PAD_TOKEN)
        self.location_y_embedding = nn.Embedding(XY_SEP_TOKEN+1, self.emb_size, padding_idx=PAD_TOKEN)
        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, location_x, location_y, loc_profile=None):

        check_ids("location_x", location_x, XY_SEP_TOKEN+1)
        check_ids("location_y", location_y, XY_SEP_TOKEN+1)
        
        location_x_emb = self.location_x_embedding(location_x)
        location_y_emb = self.location_y_embedding(location_y)
        
        if self.add_loc_profile:
            assert loc_profile is not None, "Must have location profile."
            loc_profile_emb = self.loc_profile_layer(loc_profile)
            spatial_emb = torch.cat([location_x_emb, location_y_emb, loc_profile_emb], dim=-1)
        else:
            spatial_emb = torch.cat([location_x_emb, location_y_emb], dim=-1)
            
        spatial_emb = self.spatial_proj(spatial_emb)
        return spatial_emb


class InputEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, add_trip_seq=True, add_loc_profile=False):
        super(InputEncoding, self).__init__()

        self.temporal_encoder = TemporalEncoding(emb_size, dropout)
        self.spatial_encoder = SpatialEncoding(emb_size, dropout, add_loc_profile=add_loc_profile)

        self.add_trip_seq = add_trip_seq
        if self.add_trip_seq:
            self.trip_seq_embedding = nn.Embedding(TIME_SEP_TOKEN+1, emb_size, padding_idx=PAD_TOKEN)
             
        self.input_norm = nn.LayerNorm(emb_size)
        self.input_dropout = nn.Dropout(dropout)
        self.input_mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, day, time, dow, timedelta, location_x, location_y, loc_profile=None, trip_seq=None):

        temporal_emb = self.temporal_encoder(day, time, dow, timedelta)
        spatial_emb = self.spatial_encoder(location_x, location_y, loc_profile)
        emb = temporal_emb + spatial_emb
        
        if self.add_trip_seq:
            assert trip_seq is not None
            trip_seq_emb = self.trip_seq_embedding(trip_seq)
            emb += trip_seq_emb
            
        emb = self.input_norm(emb)
        emb = emb + self.input_mlp(self.input_dropout(emb))
        return emb


### Reference Code from
### https://github.com/he-h/ST-MoE-BERT/blob/main/model.py
###
class ExpertLayer(nn.Module):
    """
    Defines a single expert layer as part of a mixture of experts, using GELU activation and linear transformation.

    Attributes:
        fc (nn.Linear): Fully connected layer that transforms input to output size.
        activation (nn.GELU): Gaussian Error Linear Unit activation function.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass of the expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output tensor after the linear transformation.
        """
        return self.activation(self.fc(x))

class MixtureOfExperts(nn.Module):
    """
    Implements a mixture of experts layer, where each input is dynamically routed to multiple expert layers based on learned gating.

    Attributes:
        experts (nn.ModuleList): A list of expert layers.
        gate (nn.Linear): Gating mechanism to determine the contribution of each expert to the output.
    """
    def __init__(self, num_experts, emb_size):
        super().__init__()
        self.experts = nn.ModuleList([ExpertLayer(emb_size, emb_size) for _ in range(num_experts)])
        self.gate = nn.Linear(emb_size, num_experts)

    def forward(self, x):
        """
        Forward pass through the mixture of experts. Computes a weighted sum of expert outputs based on gating.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The combined output of the experts, weighted by the gating mechanism.
        """
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.view(batch_size * seq_len, -1)  # Flatten input for processing by experts

        # Process input through all experts
        expert_outputs = torch.stack([expert(x_reshaped) for expert in self.experts], dim=1)
        
        # Compute gating probabilities and apply to expert outputs
        gate_logits = self.gate(x_reshaped)
        gate_probs = nn.functional.softmax(gate_logits, dim=1)
        output = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)

        output = output.view(batch_size, seq_len, -1)  # Reshape output to match input dimensions
        return output




