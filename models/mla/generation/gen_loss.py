import torch
import torch.nn as nn
import numpy as np


def chamfer_distance(pred, gt):
    # pred/gt: [B, N, 3]
    dist = torch.cdist(pred, gt)  # [B, N, M]
    loss = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
    return loss

def chamfer_distance_l2(pred, gt):
    # pred: (B, N1, 3), gt: (B, N2, 3)
    dist_matrix = torch.cdist(pred, gt)  # (B, N1, N2)
    forward_dist = dist_matrix.min(dim=2)[0].mean(dim=1)  # (B,)
    backward_dist = dist_matrix.min(dim=1)[0].mean(dim=1)  # (B,)
    chamfer_dist = forward_dist + backward_dist  # (B,)
    return chamfer_dist.mean()

def earth_movers_distance(pred, gt):
    # Flatten batch dimension for pairwise distance computation
    B, N, _ = pred.shape
    pred_flat = pred.view(-1, 3)  # [B*N, 3]
    gt_flat = gt.view(-1, 3)      # [B*N, 3]
    
    dist_matrix = torch.cdist(pred_flat, gt_flat)  # [B*N, B*N]
    
    dist_matrix = dist_matrix.view(B, N, B, N)
    
    emd_loss = 0.0
    for i in range(B):
        # Hungarian algorithm would be better but computationally expensive
        # Here we use a simplified approximation
        row_min = dist_matrix[i, :, i, :].min(dim=1)[0].mean()
        col_min = dist_matrix[i, :, i, :].min(dim=0)[0].mean()
        emd_loss += (row_min + col_min) / 2
        
    return emd_loss / B