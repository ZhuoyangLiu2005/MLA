import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

def visualize_reconstruction(reconstruction_outputs, next_images, next_pointclouds, save_dir):
    """
    可视化重建结果（仅坐标重建版本）
    Args:
        reconstruction_outputs: 重建输出字典（需包含'image_reconstruction'和'pointcloud_coord_reconstruction'）
        next_images: GT图像 [B, C, H, W]
        next_pointclouds: GT点云 [B, N, 3]（仅坐标）
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_idx = 0 
    
    if 'image_reconstruction' in reconstruction_outputs and next_images is not None:
        image_pred = reconstruction_outputs['image_reconstruction'][batch_idx].detach().cpu()
        # print(image_pred.shape)
        # input()
        gt_image = next_images[batch_idx].detach().cpu().float().permute(1, 2, 0).numpy()
        
        patch_size = 42  
        num_patches_side = 672 // patch_size  
        patches = image_pred.view(num_patches_side, num_patches_side, patch_size, patch_size, 3)
        # print(patches.shape)
        # input()
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
        # print(coord_pred.shape)
        # input()
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

# visualize_reconstruction(reconstruction_outputs, next_images, next_pointclouds, "recon_vis")