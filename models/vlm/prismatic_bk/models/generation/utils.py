import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def images_to_patches(images, patch_size=42):
    B, C, H, W = images.shape

    assert C == 3, f"Expected 3 channels (RGB), got {C}"
    assert H == W == 672, f"Expected 672x672 image, got {H}x{W}"

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # [B, C=3, num_patches_h=16, num_patches_w=16, patch_h=42, patch_w=42]
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size) # [B, C=3, num_patches=256, patch_h=42, patch_w=42]
    patches = patches.permute(0, 2, 1, 3, 4).contiguous() # [B, num_patches=256, C=3, patch_h=42, patch_w=42]
    patches = patches.view(B, -1, C * patch_size * patch_size) # [B, num_patches=256, C*patch_h*patch_w=5292]

    return patches
    
def patches_to_images(patches, patch_size=42):
    B, num_patches, patch_dim = patches.shape
    
    expected_patch_dim = 3 * patch_size * patch_size
    assert patch_dim == expected_patch_dim, f"Expected patch_dim={expected_patch_dim}, got {patch_dim}"
    assert num_patches == 256, f"Expected 256 patches, got {num_patches}"
    
    patches = patches.view(B, num_patches, 3, patch_size, patch_size) # [B, 256, 3, 42, 42]
    num_patches_h = num_patches_w = int(num_patches ** 0.5)  # 16
    patches = patches.view(B, num_patches_h, num_patches_w, 3, patch_size, patch_size)
    images = patches.permute(0, 3, 1, 4, 2, 5).contiguous() # [B, 3, 16, 42, 16, 42]
    images = images.view(B, 3, num_patches_h * patch_size, num_patches_w * patch_size) # [B, 3, 672, 672]
    
    return images

def dilate_mask(mask, kernel_size):
    """Dilate a boolean mask of shape [B, H, W] using max_pool2d safely."""
    # mask: [B, H, W] (bool)
    assert mask.dim() == 3, "Expected mask shape [B,H,W]"
    padding = (kernel_size - 1) // 2
    # Make it 4D: [B, C=1, H, W]
    m = mask.float().unsqueeze(1)  # [B,1,H,W]
    dil = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=padding)
    dil = (dil > 0.0).squeeze(1)   # [B,H,W] bool
    return dil

def create_roi_mask_from_indices(patch_indices: torch.Tensor) -> torch.Tensor:

    batch_size = patch_indices.shape[0]
    patch_grid_size = 16 

    roi_mask = torch.zeros(
        batch_size, 
        patch_grid_size, 
        patch_grid_size, 
        dtype=torch.bool, 
        device=patch_indices.device
    )

    batch_idx = torch.arange(batch_size, device=patch_indices.device).view(batch_size, 1)
    y_coords = patch_indices[..., 0] 
    x_coords = patch_indices[..., 1] 
    roi_mask[batch_idx, y_coords, x_coords] = True

    return roi_mask

