import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .utils import dilate_mask


class ImageReconstructionModule(nn.Module):
    def __init__(
        self,
        token_size: int = 4096,
        num_recon_queries: int = 64,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        image_patch_size: int = 42,
        use_roi: bool = True,
        roi_dilation_kernel_size: int = 3,
        recon_delta_clip: float = 5.0,
        max_patch_shift_pixels: int = 8,
        use_patch_offset: bool = True,
    ):
        super().__init__()
        
        # Configuration
        self.token_size = token_size
        self.num_recon_queries = num_recon_queries
        self.image_patch_size = image_patch_size
        self.use_roi = use_roi
        self.roi_dilation_kernel_size = roi_dilation_kernel_size
        self.recon_delta_clip = recon_delta_clip
        self.max_patch_shift_pixels = max_patch_shift_pixels
        self.use_patch_offset = use_patch_offset
        self.image_num_patches = 256
        
        # Learnable parameters
        self.image_recon_queries = nn.Parameter(
            torch.zeros(1, self.num_recon_queries, token_size)
        )
        self.mae_mask_token = nn.Parameter(torch.zeros(1, 1, token_size))
        self.mae_pos_embed = nn.Parameter(torch.zeros(1, self.image_num_patches, token_size))
        
        # Intent decoder for high-level feature extraction
        intent_decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_size, 
            nhead=decoder_heads, 
            dim_feedforward=token_size * 2,
            dropout=0.1, 
            activation='gelu', 
            batch_first=True
        )
        self.intent_decoder = nn.TransformerDecoder(intent_decoder_layer, num_layers=2)
        
        # MAE decoder for patch reconstruction
        mae_decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_size,
            nhead=decoder_heads,
            dim_feedforward=token_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.mae_decoder = nn.TransformerDecoder(mae_decoder_layer, num_layers=decoder_layers)
        
        # Prediction heads
        patch_dim = self.image_patch_size ** 2 * 3
        self.mae_patch_norm = nn.LayerNorm(token_size)
        self.mae_delta_head = nn.Linear(token_size, patch_dim)
        self.mae_alpha_head = nn.Linear(token_size, 1)
        self.mae_offset_head = nn.Linear(token_size, 2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all learnable parameters"""
        # Initialize query tokens and embeddings
        nn.init.normal_(self.image_recon_queries, std=0.02)
        nn.init.normal_(self.mae_mask_token, std=0.02)
        self._init_pos_embed(self.mae_pos_embed)
        
        # Initialize prediction heads
        nn.init.normal_(self.mae_delta_head.weight, std=0.02)
        if self.mae_delta_head.bias is not None:
            nn.init.constant_(self.mae_delta_head.bias, 0.0)
        
        # Alpha head: bias negative so initially model prefers copying current patch
        nn.init.normal_(self.mae_alpha_head.weight, std=0.02)
        if self.mae_alpha_head.bias is not None:
            nn.init.constant_(self.mae_alpha_head.bias, -3.0)
        
        # Offset head: small initialization
        nn.init.normal_(self.mae_offset_head.weight, std=0.001)
        if self.mae_offset_head.bias is not None:
            nn.init.constant_(self.mae_offset_head.bias, 0.0)
    
    def _init_pos_embed(self, pos_embed):
        nn.init.normal_(pos_embed, std=0.02)
    
    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        current_image_features: torch.Tensor,
        current_images_patches: torch.Tensor,
        roi_mask_2d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = llm_hidden_states.shape[0]
        
        # Extract intent features using query-based attention
        intent_queries = self.image_recon_queries.expand(batch_size, -1, -1)
        intent_features = self.intent_decoder(tgt=intent_queries, memory=llm_hidden_states)
        
        # Prepare ROI mask
        if self.use_roi:
            reconstruction_roi_mask_2d = dilate_mask(roi_mask_2d, self.roi_dilation_kernel_size)
            reconstruction_roi_mask = reconstruction_roi_mask_2d.view(batch_size, -1)
        else:
            reconstruction_roi_mask = torch.ones(
                (batch_size, current_images_patches.shape[1]),
                dtype=torch.bool,
                device=current_images_patches.device
            )
        
        # Prepare decoder input: mask tokens at ROI positions
        decoder_input_tokens = current_image_features.clone()
        mask_token_vec = self.mae_mask_token.view(-1)
        decoder_input_tokens[reconstruction_roi_mask] = mask_token_vec
        decoder_input_tokens = decoder_input_tokens + self.mae_pos_embed
        
        # Run MAE decoder
        reconstructed_features = self.mae_decoder(
            tgt=decoder_input_tokens, 
            memory=intent_features
        )
        
        # Predict delta, alpha, and offset for all patches
        features_flat = reconstructed_features.view(-1, reconstructed_features.shape[-1])
        features_norm = self.mae_patch_norm(features_flat)
        
        delta_all = self.mae_delta_head(features_norm)
        alpha_all = torch.sigmoid(self.mae_alpha_head(features_norm).squeeze(-1))
        offset_all = self.mae_offset_head(features_norm)
        
        # Reshape predictions
        B, num_patches, _ = reconstructed_features.shape
        patch_dim = self.image_patch_size ** 2 * 3
        delta_all = delta_all.view(B, num_patches, patch_dim)
        alpha_all = alpha_all.view(B, num_patches)
        offset_all = offset_all.view(B, num_patches, 2)
        
        # Apply scaling and clipping
        delta_all = torch.tanh(delta_all) * self.recon_delta_clip
        offset_all = torch.tanh(offset_all) * float(self.max_patch_shift_pixels)
        
        # Generate final reconstruction
        reconstructed_patches = self._generate_reconstructed_patches(
            current_images_patches, delta_all, alpha_all, offset_all, reconstruction_roi_mask
        )
        
        return {
            'image_reconstruction': reconstructed_patches,
            'reconstruction_roi_mask': reconstruction_roi_mask,
            'delta_all': delta_all,
            'alpha_all': alpha_all,
            'offset_all': offset_all,
        }
    
    def _generate_reconstructed_patches(
        self,
        curr_patches: torch.Tensor,
        delta_all: torch.Tensor,
        alpha_all: torch.Tensor,
        offset_all: torch.Tensor,
        reconstruction_roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate final reconstructed patches with warping and blending"""
        B, num_patches, patch_dim = curr_patches.shape
        C = 3
        ps = self.image_patch_size
        
        # Reshape for image operations
        curr_patches_img = curr_patches.view(B * num_patches, C, ps, ps)
        offset_all_img = offset_all.view(B * num_patches, 2)
        
        # Apply patch offset warping if enabled
        if self.use_patch_offset:
            tx = offset_all_img[:, 0]
            ty = offset_all_img[:, 1]
            tx_norm = 2.0 * tx / float(ps - 1)
            ty_norm = 2.0 * ty / float(ps - 1)
            
            affines = torch.zeros(B * num_patches, 2, 3, device=offset_all_img.device, dtype=offset_all_img.dtype)
            affines[:, 0, 0] = 1.0
            affines[:, 1, 1] = 1.0
            affines[:, 0, 2] = tx_norm
            affines[:, 1, 2] = ty_norm
            
            # Handle potential dtype issues with grid_sample
            curr_patches_img_f = curr_patches_img.float()
            grid = F.affine_grid(affines.float(), size=(B * num_patches, C, ps, ps), align_corners=True)
            warped = F.grid_sample(curr_patches_img_f, grid, mode='bilinear', padding_mode='border', align_corners=True)
            warped = warped.to(curr_patches_img.dtype)
        else:
            warped = curr_patches_img
        
        # Apply delta predictions
        delta_img = delta_all.view(B * num_patches, C, ps, ps)
        gen_weight = 0.95
        pure_pred = delta_img
        residual_pred = curr_patches_img + delta_img
        roi_pred = (1 - gen_weight) * residual_pred + gen_weight * pure_pred
        non_roi_pred = warped + delta_img
        
        # Apply ROI-specific predictions
        roi_mask_flat = reconstruction_roi_mask.view(B * num_patches, 1, 1, 1)
        predicted_img_patches = torch.where(roi_mask_flat, roi_pred, non_roi_pred)
        
        # Alpha blending
        alpha_all = torch.where(
            reconstruction_roi_mask,
            torch.ones_like(alpha_all),
            alpha_all
        )
        alpha_img = alpha_all.view(B * num_patches, 1, 1, 1)
        blended = alpha_img * predicted_img_patches + (1.0 - alpha_img) * curr_patches_img
        
        # Reshape back to patch format
        return blended.view(B, num_patches, -1)
    
    
    
    
class PointCloudReconstructionModule(nn.Module):
    def __init__(
        self,
        token_size: int = 4096,
        num_recon_queries: int = 64,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        pointcloud_dim: int = 768,
        num_patches: int = 1024,
    ):
        super().__init__()
        
        # Configuration
        self.token_size = token_size
        self.num_recon_queries = num_recon_queries
        self.pointcloud_dim = pointcloud_dim
        self.num_patches = num_patches
        
        # Learnable parameters
        self.pointcloud_recon_queries = nn.Parameter(
            torch.zeros(1, num_recon_queries, token_size)
        )
        self.pointcloud_mask_tokens = nn.Parameter(
            torch.zeros(1, num_patches, pointcloud_dim)
        )
        self.pointcloud_recon_pos_embed = nn.Parameter(
            torch.zeros(1, num_recon_queries + num_patches, pointcloud_dim)
        )
        
        # Transformer decoder for point cloud reconstruction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_size,
            nhead=decoder_heads,
            dim_feedforward=token_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.pointcloud_recon_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        
        # Projection and prediction layers
        self.pointcloud_recon_projector = nn.Linear(token_size, pointcloud_dim)
        self.pointcloud_coord_predictor = nn.Sequential(
            nn.LayerNorm(pointcloud_dim),
            nn.Linear(pointcloud_dim, 3)  # XYZ coordinates
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all learnable parameters"""
        nn.init.normal_(self.pointcloud_recon_queries, std=0.02)
        nn.init.normal_(self.pointcloud_mask_tokens, std=0.02)
        self._init_pos_embed(self.pointcloud_recon_pos_embed)
    
    def _init_pos_embed(self, pos_embed):
        """Initialize position embeddings"""
        nn.init.normal_(pos_embed, std=0.02)
    
    def forward(
        self,
        llm_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = llm_hidden_states.shape[0]
        
        # Extract point cloud reconstruction features
        pc_queries = self.pointcloud_recon_queries.expand(batch_size, -1, -1)
        pc_recon_features = self.pointcloud_recon_decoder(
            tgt=pc_queries,
            memory=llm_hidden_states
        )
        
        # Project to point cloud feature space
        pc_recon_proj = self.pointcloud_recon_projector(pc_recon_features)
        
        # Combine with mask tokens and add positional embeddings
        pc_mask_tokens = self.pointcloud_mask_tokens.expand(batch_size, -1, -1)
        pc_decoder_input = torch.cat([pc_recon_proj, pc_mask_tokens], dim=1)
        pc_decoder_input = pc_decoder_input + self.pointcloud_recon_pos_embed
        
        # Extract features for coordinate prediction
        pc_mask_features = pc_decoder_input[:, self.num_recon_queries:, :]
        
        # Predict 3D coordinates
        pc_coord_pred = self.pointcloud_coord_predictor(pc_mask_features)
        
        return {
            'pointcloud_coord_reconstruction': pc_coord_pred,
            'pointcloud_features': pc_recon_features,
        }
        


class MultimodalReconstructionManager(nn.Module):
    def __init__(
        self,
        token_size: int = 4096,
        # Image reconstruction parameters
        use_image_reconstruction: bool = True,
        num_image_recon_queries: int = 64,
        image_decoder_layers: int = 3,
        image_decoder_heads: int = 8,
        image_patch_size: int = 42,
        use_roi: bool = True,
        roi_dilation_kernel_size: int = 3,
        # Point cloud reconstruction parameters
        use_pointcloud_reconstruction: bool = False,
        num_pointcloud_recon_queries: int = 64,
        pointcloud_decoder_layers: int = 3,
        pointcloud_decoder_heads: int = 8,
        pointcloud_dim: int = 768,
        pointcloud_num_patches: int = 1024,
    ):
        super().__init__()
        
        self.use_image_reconstruction = use_image_reconstruction
        self.use_pointcloud_reconstruction = use_pointcloud_reconstruction
        
        # Initialize reconstruction modules
        if self.use_image_reconstruction:
            self.image_recon_module = ImageReconstructionModule(
                token_size=token_size,
                num_recon_queries=num_image_recon_queries,
                decoder_layers=image_decoder_layers,
                decoder_heads=image_decoder_heads,
                image_patch_size=image_patch_size,
                use_roi=use_roi,
                roi_dilation_kernel_size=roi_dilation_kernel_size,
            )
        
        if self.use_pointcloud_reconstruction:
            self.pointcloud_recon_module = PointCloudReconstructionModule(
                token_size=token_size,
                num_recon_queries=num_pointcloud_recon_queries,
                decoder_layers=pointcloud_decoder_layers,
                decoder_heads=pointcloud_decoder_heads,
                pointcloud_dim=pointcloud_dim,
                num_patches=pointcloud_num_patches,
            )
    
    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        current_image_features: Optional[torch.Tensor] = None,
        current_images_patches: Optional[torch.Tensor] = None,
        roi_mask_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        reconstruction_outputs = {}
        
        if self.use_image_reconstruction and current_image_features is not None:
            image_outputs = self.image_recon_module(
                llm_hidden_states=llm_hidden_states,
                current_image_features=current_image_features,
                current_images_patches=current_images_patches,
                roi_mask_2d=roi_mask_2d,
            )
            reconstruction_outputs.update(image_outputs)
        
        if self.use_pointcloud_reconstruction:
            pointcloud_outputs = self.pointcloud_recon_module(
                llm_hidden_states=llm_hidden_states,
            )
            reconstruction_outputs.update(pointcloud_outputs)
        
        return reconstruction_outputs
    
    def get_module_keys(self) -> list:
        """Get all module keys for checkpoint saving/loading"""
        keys = []
        if self.use_image_reconstruction:
            keys.append("image_recon_module")
        if self.use_pointcloud_reconstruction:
            keys.append("pointcloud_recon_module")
        return keys