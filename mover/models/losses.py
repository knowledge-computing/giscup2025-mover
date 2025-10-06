import torch
import torch.nn.functional as F
from torch import nn

from entmax import Entmax15Loss, entmax15


def logit_mse(logits_a, logits_b):
    """MSE in logit space (common, stable alternative to KL/JS)."""
    return F.mse_loss(logits_a, logits_b, reduction="mean")


def get_loss(output, batch, activation='softmax', losses=[]):

    out_loss, total_loss = {}, 0
    keep = (batch['d_tgt'] != 0)

    ##########################################################################################
    if 'ce' in losses:
        criterion = nn.CrossEntropyLoss()
        if output.get("pred_x") is not None and output.get("pred_y") is not None:
            pred_x, pred_y = output["pred_x"], output["pred_y"]
            loss = criterion(pred_x[keep], batch['label_x_tgt'][keep])
            loss += criterion(pred_y[keep], batch['label_y_tgt'][keep])

        if output.get("pred_x1") is not None and output.get("pred_y1") is not None:
            pred_x, pred_y = output["pred_x1"], output["pred_y1"]
            loss = criterion(pred_x[keep], batch['label_x_tgt'][keep])
            loss += criterion(pred_y[keep], batch['label_y_tgt'][keep])
            pred_x, pred_y = output["pred_x2"], output["pred_y2"]
            loss += criterion(pred_x[keep], batch['label_x_tgt'][keep])
            loss += criterion(pred_y[keep], batch['label_y_tgt'][keep])

            # loss += logit_mse(output["pred_x1"][keep], output["pred_x2"][keep].detach()) * 0.1
            # loss += logit_mse(output["pred_y2"][keep], output["pred_y1"][keep].detach()) * 0.1
            
            pred_x = (output["pred_x1"] + output["pred_x2"]) / 2
            pred_y = (output["pred_y1"] + output["pred_y2"]) / 2
            loss += criterion(pred_x[keep], batch['label_x_tgt'][keep]) * 0.01
            loss += criterion(pred_y[keep], batch['label_y_tgt'][keep]) * 0.01

            # pred_x = (output["pred_x1"] + output["pred_x2"]) / 2
            # pred_y = (output["pred_y1"] + output["pred_y2"]) / 2
            # loss = criterion(pred_x[keep], batch['label_x_tgt'][keep])
            # loss += criterion(pred_y[keep], batch['label_y_tgt'][keep])
            
        if output.get("pred_2d") is not None:
            gt = batch['label_y_tgt'] * 200 + batch['label_x_tgt']
            loss += criterion(output["pred_2d"][keep], gt[keep])
                        
        out_loss['ce'] = loss.detach().item()
        total_loss += loss

    ##########################################################################################
    if 'binary' in losses:
        logits = output["pred_sep"].squeeze(-1).float()
        labels = batch["label_sep_token"].to(dtype=logits.dtype)
        pos_weight = ((keep.sum() - labels[keep].sum()) / (labels[keep].sum() + 1e-6))
        pos_weight = torch.clamp(pos_weight, max=10.0)          # limit scale
        pos_weight = pos_weight.to(dtype=logits.dtype, device=logits.device)
        loss_per_token = F.binary_cross_entropy_with_logits(logits, labels, reduction='none', pos_weight=pos_weight) 
        loss = (loss_per_token * keep.float()).sum() / (keep.float().sum().clamp_min(1.0))
        out_loss['binary'] = loss.detach().item()
        total_loss += loss * 0.1

    ##########################################################################################
    if 'entropy' in losses:
        def entropy_bonus(logits):
            p = torch.softmax(logits, dim=-1) 
            ent = -(p * (p.clamp_min(1e-9)).log()).sum(-1)  
            return ent.mean()
            
        entropy_x = entropy_bonus(pred_x[keep])
        entropy_y = entropy_bonus(pred_y[keep])
        out_loss['entropy'] = (entropy_x + entropy_y).detach().item()
        total_loss = total_loss - 0.01 * (entropy_x + entropy_y)

    # ##########################################################################################
    # if 'soft_ce' in losses:
    #     if output.get("pred_x") is not None and output.get("pred_y") is not None:
    #         loss = soft_ce_1d(output["pred_x"], batch['label_x_tgt'], keep, sigma=1.8, activation=activation)
    #         loss += soft_ce_1d(output["pred_y"], batch['label_y_tgt'], keep, sigma=1.8, activation=activation)
            
    #     if output.get("pred_2d") is not None:
    #         loss = soft_ce_2d(output["pred_2d"], batch['label_y_tgt'], batch['label_x_tgt'], keep, sigma=1.8, activation=activation)
            
    #     out_loss['soft_ce'] = loss.detach().item()
    #     total_loss += loss

    return out_loss, total_loss


def gaussian_1d_centers(idx, size, sigma, device, clip_eps: float = 0.0):
    t = torch.arange(size, device=device, dtype=torch.float32)
    diff = t.unsqueeze(0) - idx.unsqueeze(1)          
    g = torch.exp(- diff ** 2 / (2 * sigma ** 2)) # N x 200
    if clip_eps > 0:
        g = torch.where(g >= clip_eps, g, torch.zeros_like(g))
    g = g / (g.sum(dim=1, keepdim=True) + 1e-12)
    return g


# def soft_ce_1d(logits, gt, keep, sigma=1.5, activation="softmax"):
#     B, L, K = logits.shape
#     device = logits.device

#     flat_logits = logits.reshape(-1, K)
#     flat_keep = keep.reshape(-1)
#     flat_gt = gt.reshape(-1)

#     valid_logits = flat_logits[flat_keep]
#     if activation == "softmax":
#         logp = F.log_softmax(valid_logits, dim=-1).view(-1, K)
#         eff_clip_eps = 0.0
#     elif activation == "entmax15":
#         p = entmax15(valid_logits, dim=-1).view(-1, K)
#         print(p)
#         logp = torch.log(p.clamp_min(1e-12))
#         eff_clip_eps = 1e-3
#     else:
#         raise ValueError("activation must be 'softmax' or 'entmax15'")

#     g = flat_gt[flat_keep]
#     g = gaussian_1d_centers(g, K, sigma, device, clip_eps=eff_clip_eps)
#     return -(g * logp).sum(dim=-1).mean()


# def soft_ce_2d(logits, rows, cols, keep, sigma=1.5, activation="softmax"):
#     H = W = 200
#     B, L, HW = logits.shape
#     assert HW == H * W
#     device = logits.device

#     valid_logits = logits[keep]
#     if activation == "softmax":
#         logp = F.log_softmax(valid_logits, dim=-1).view(-1, H, W)
#         eff_clip_eps = 0.0
#     elif activation == "entmax15":
#         p = entmax15(valid_logits, dim=-1).view(-1, H, W)
#         logp = torch.log(p.clamp_min(1e-12))
#         eff_clip_eps = 1e-3
#     else:
#         raise ValueError("activation must be 'softmax' or 'entmax15'")

#     r0 = rows[keep].float()
#     c0 = cols[keep].float()
#     gr = gaussian_1d_centers(r0, H, sigma, device, clip_eps=eff_clip_eps)      # [N, H]
#     gc = gaussian_1d_centers(c0, W, sigma, device, clip_eps=eff_clip_eps)      # [N, W]

#     # Compute sum_{i,j} gr[i] * logp[i,j] * gc[j] via two batched matmuls
#     # Step 1: T = gr^T @ logp  => [N, 1, W]
#     T = torch.bmm(gr.unsqueeze(1), logp)                  # [N, 1, W]
#     # Step 2: s = T @ gc => [N, 1, 1]
#     s = torch.bmm(T, gc.unsqueeze(2))                     # [N, 1, 1]
#     # Cross-entropy with soft target = - E_{G}[log p]
#     loss = -s.view(-1).mean()            
#     return loss

