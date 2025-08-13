import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import Axes3D
from .utils import images_to_patches, patches_to_images


def visualize_reconstruction_simple(reconstruction_outputs, next_images, next_pointclouds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    batch_idx = 0 
    
    if 'image_reconstruction' in reconstruction_outputs and next_images is not None:
        image_pred = reconstruction_outputs['image_reconstruction'][batch_idx].detach().cpu()
        gt_image = next_images[batch_idx].detach().cpu().float().permute(1, 2, 0).numpy()
        
        patch_size = 42  
        num_patches_side = 672 // patch_size  
        patches = image_pred.view(num_patches_side, num_patches_side, patch_size, patch_size, 3)
        recon_image = patches.permute(0, 2, 1, 3, 4).contiguous()
        recon_image = recon_image.view(num_patches_side * patch_size, num_patches_side * patch_size, 3).float().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(gt_image)
        axs[0].set_title("GT Image")
        axs[1].imshow(np.clip(recon_image, 0, 1))
        axs[1].set_title("Reconstructed Image")
        plt.savefig(os.path.join(save_dir, f"image_recon.png"))
        plt.close()
    
    if 'pointcloud_coord_reconstruction' in reconstruction_outputs and next_pointclouds is not None:
        coord_pred = reconstruction_outputs['pointcloud_coord_reconstruction'][batch_idx].detach().cpu().float().numpy()
        gt_points = next_pointclouds[batch_idx, :, :3].detach().cpu().float().numpy()

        fig = plt.figure(figsize=(15, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='blue', s=2)
        ax1.set_title("GT Point Cloud")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(coord_pred[:, 0], coord_pred[:, 1], coord_pred[:, 2], c='red', s=2)
        ax2.set_title("Reconstructed Point Cloud")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.savefig(os.path.join(save_dir, f"pointcloud_recon.png"))
        plt.close()


def visualize_reconstruction_diff(reconstruction_outputs, current_images, next_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_patch_size=42
    if current_images is None or 'image_reconstruction' not in reconstruction_outputs:
        return
    
    batch_size = current_images.shape[0]
    for b in range(min(batch_size, 1)):  # 只可视化前4个样本
        current_img = current_images[:, :3, :, :][b].cpu().float().numpy().transpose(1, 2, 0)  # CHW -> HWC
        next_img = next_images[b].cpu().float().numpy().transpose(1, 2, 0) if next_images is not None else None

        reconstructed_patches = reconstruction_outputs['image_reconstruction'][b:b+1]  # [1, N_patches, patch_dim]
        reconstructed_img = patches_to_images(reconstructed_patches, image_patch_size)[0]  # [C, H, W]
        reconstructed_img = reconstructed_img.cpu().float().detach().numpy().transpose(1, 2, 0)  # CHW -> HWC
        
        if 'change_masks' in reconstruction_outputs:
            change_mask = reconstruction_outputs['change_masks'][b].cpu().float().detach()  # [N_patches, 1]
            change_mask_patches = change_mask.reshape(1, -1, 1).repeat_interleave(image_patch_size**2 * 3, dim=2)
            change_mask_img = patches_to_images(change_mask_patches, image_patch_size)[0, 0].cpu().float().numpy()
        else:
            change_mask_img = None

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        axes[0].imshow(current_img)
        axes[0].set_title('Current Frame')
        axes[0].axis('off')
        
        if next_img is not None:
            axes[1].imshow(next_img)
            axes[1].set_title('Next Frame (GT)')
            axes[1].axis('off')
        
        axes[2].imshow(reconstructed_img)
        axes[2].set_title('Reconstructed Next Frame')
        axes[2].axis('off')
        
        if change_mask_img is not None:
            axes[3].imshow(change_mask_img, cmap='hot')
            axes[3].set_title('Change Mask')
            axes[3].axis('off')
        
        # 差异图
        if next_img is not None:
            diff_img = np.abs(reconstructed_img - next_img)
            axes[4].imshow(diff_img)
            axes[4].set_title('Reconstruction Error')
            axes[4].axis('off')
        
        # 当前帧和下一帧的差异
        if next_img is not None:
            real_diff = np.abs(next_img - current_img)
            axes[5].imshow(real_diff)
            axes[5].set_title('Real Change')
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_sample_{b}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
def visualize_reconstruction_rgb(reconstruction_outputs, 
                                 image_patch_size, 
                                 next_images, 
                                 save_dir
                            ):
    """8-subplot layout with Target/Recon RGB + their separate channels"""
    os.makedirs(save_dir, exist_ok=True)
    if 'image_reconstruction' in reconstruction_outputs and next_images is not None:
        # Convert patches to full images
        patch_pred = reconstruction_outputs['image_reconstruction']  # [B, num_patches, patch_dim]
        reconstructed_images = patches_to_images(patch_pred, image_patch_size)  # [B, 3, 672, 672]

        # Clamp values to [0,1] range
        # reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
        # next_images = torch.clamp(next_images, 0, 1)
        
        batch_size = min(1, reconstructed_images.shape[0])  # Only visualize first sample
        
        for i in range(batch_size):
            # Create 2x4 subplots
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Image Reconstruction Comparison', fontsize=16)

            # --- Row 1: Target ---
            # Target RGB
            target_img = next_images[i].cpu().float().detach().permute(1, 2, 0).numpy()
            axes[0, 0].imshow(target_img)
            axes[0, 0].set_title('Target (RGB)')
            axes[0, 0].axis('off')

            # Target R/G/B Channels
            target_tensor = next_images[i].cpu().float().detach()
            axes[0, 1].imshow(target_tensor[0], cmap='Reds', vmin=0, vmax=1)
            axes[0, 1].set_title('Target (Red)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(target_tensor[1], cmap='Greens', vmin=0, vmax=1)
            axes[0, 2].set_title('Target (Green)')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(target_tensor[2], cmap='Blues', vmin=0, vmax=1)
            axes[0, 3].set_title('Target (Blue)')
            axes[0, 3].axis('off')

            # --- Row 2: Reconstructed ---
            # Reconstructed RGB
            recon_img = reconstructed_images[i].cpu().float().detach().permute(1, 2, 0).numpy()
            axes[1, 0].imshow(recon_img)
            axes[1, 0].set_title('Recon (RGB)')
            axes[1, 0].axis('off')

            # Reconstructed R/G/B Channels
            recon_tensor = reconstructed_images[i].cpu().float().detach()
            axes[1, 1].imshow(recon_tensor[0], cmap='Reds', vmin=0, vmax=1)
            axes[1, 1].set_title('Recon (Red)')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(recon_tensor[1], cmap='Greens', vmin=0, vmax=1)
            axes[1, 2].set_title('Recon (Green)')
            axes[1, 2].axis('off')
            
            axes[1, 3].imshow(recon_tensor[2], cmap='Blues', vmin=0, vmax=1)
            axes[1, 3].set_title('Recon (Blue)')
            axes[1, 3].axis('off')

            # Save figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f'reconstruction_8subplots_{i}.png'), 
                dpi=150, 
                bbox_inches='tight'
            )
            plt.close()

            # Optional: Save individual images
            save_image(next_images[i], os.path.join(save_dir, f'target_{i}.png'))
            save_image(reconstructed_images[i], os.path.join(save_dir, f'reconstructed_{i}.png'))
            
        # print(f"Visualization saved to {save_dir}")