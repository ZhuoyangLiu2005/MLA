import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import Axes3D
from .utils import images_to_patches, patches_to_images


def visualize_generation_simple(generation_outputs, next_images, next_pointclouds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    batch_idx = 0 
    image_patch_size=42
    if 'image_generation' in generation_outputs and next_images is not None:
        patch_pred = generation_outputs['image_generation']  # [B, num_patches, patch_dim]
        generated_images = patches_to_images(patch_pred, image_patch_size)  # [B, 3, 672, 672]
        gt_image = next_images[batch_idx].detach().cpu().float().permute(1, 2, 0).numpy()
        
        gen_image = generated_images[batch_idx].cpu().float().detach().permute(1, 2, 0).numpy()
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(gt_image)
        axs[0].set_title("GT Image")
        axs[1].imshow(gen_image)
        axs[1].set_title("Generated Image")
        plt.savefig(os.path.join(save_dir, f"image_gen.png"))
        plt.close()
    
    if 'pointcloud_coord_generation' in generation_outputs and next_pointclouds is not None:
        coord_pred = generation_outputs['pointcloud_coord_generation'][batch_idx].detach().cpu().float().numpy()
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
        ax2.set_title("Generated Point Cloud")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.savefig(os.path.join(save_dir, f"pointcloud_gen.png"))
        plt.close()

        