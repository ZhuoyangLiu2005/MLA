import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from timm.models.layers import DropPath, trunc_normal_
from .utils import dilate_mask


class FPSSampling(nn.Module):
    
    def __init__(self, num_group: int):
        super().__init__()
        self.num_group = num_group
    
    def fps_sampling(self, xyz, num_samples):
        B, N, _ = xyz.shape
        device = xyz.device

        centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(num_samples):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        sampled_points = xyz[torch.arange(B).unsqueeze(1), centroids]
        return sampled_points
    
    def forward(self, xyz):
        centers = self.fps_sampling(xyz, self.num_group)
        return centers


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, pos=None):
        if pos is not None:
            x_norm = self.norm1(x + pos)
        else:
            x_norm = self.norm1(x)
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ImageGenerationModule(nn.Module):
    def __init__(
        self,
        token_size: int = 4096,
        num_gen_queries: int = 64,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        image_patch_size: int = 42,
        use_roi: bool = True,
        roi_dilation_kernel_size: int = 3,
        gen_delta_clip: float = 5.0,
        max_patch_shift_pixels: int = 8,
        use_patch_offset: bool = True,
    ):
        super().__init__()
        
        # Configuration
        self.token_size = token_size
        self.num_gen_queries = num_gen_queries
        self.image_patch_size = image_patch_size
        self.use_roi = use_roi
        self.roi_dilation_kernel_size = roi_dilation_kernel_size
        self.gen_delta_clip = gen_delta_clip
        self.max_patch_shift_pixels = max_patch_shift_pixels
        self.use_patch_offset = use_patch_offset
        self.image_num_patches = 256
        
        # Learnable parameters
        self.image_gen_queries = nn.Parameter(
            torch.zeros(1, self.num_gen_queries, token_size)
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
        
        # MAE decoder for patch generation
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
        nn.init.normal_(self.image_gen_queries, std=0.02)
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
        intent_queries = self.image_gen_queries.expand(batch_size, -1, -1)
        intent_features = self.intent_decoder(tgt=intent_queries, memory=llm_hidden_states)
        
        # Prepare ROI mask
        if self.use_roi:
            generation_roi_mask_2d = dilate_mask(roi_mask_2d, self.roi_dilation_kernel_size)
            generation_roi_mask = generation_roi_mask_2d.view(batch_size, -1)
        else:
            generation_roi_mask = torch.ones(
                (batch_size, current_images_patches.shape[1]),
                dtype=torch.bool,
                device=current_images_patches.device
            )
        
        # Prepare decoder input: mask tokens at ROI positions
        decoder_input_tokens = current_image_features.clone()
        mask_token_vec = self.mae_mask_token.view(-1)
        decoder_input_tokens[generation_roi_mask] = mask_token_vec
        decoder_input_tokens = decoder_input_tokens + self.mae_pos_embed
        
        # Run MAE decoder
        generated_features = self.mae_decoder(
            tgt=decoder_input_tokens, 
            memory=intent_features
        )
        
        # Predict delta, alpha, and offset for all patches
        features_flat = generated_features.view(-1, generated_features.shape[-1])
        features_norm = self.mae_patch_norm(features_flat)
        
        delta_all = self.mae_delta_head(features_norm)
        alpha_all = torch.sigmoid(self.mae_alpha_head(features_norm).squeeze(-1))
        offset_all = self.mae_offset_head(features_norm)
        
        # Reshape predictions
        B, num_patches, _ = generated_features.shape
        patch_dim = self.image_patch_size ** 2 * 3
        delta_all = delta_all.view(B, num_patches, patch_dim)
        alpha_all = alpha_all.view(B, num_patches)
        offset_all = offset_all.view(B, num_patches, 2)
        
        # Apply scaling and clipping
        delta_all = torch.tanh(delta_all) * self.gen_delta_clip
        offset_all = torch.tanh(offset_all) * float(self.max_patch_shift_pixels)
        
        # Generate final generation
        generated_patches = self._generate_generated_patches(
            current_images_patches, delta_all, alpha_all, offset_all, generation_roi_mask
        )
        
        return {
            'image_generation': generated_patches,
            'generation_roi_mask': generation_roi_mask,
            'delta_all': delta_all,
            'alpha_all': alpha_all,
            'offset_all': offset_all,
        }
    
    def _generate_generated_patches(
        self,
        curr_patches: torch.Tensor,
        delta_all: torch.Tensor,
        alpha_all: torch.Tensor,
        offset_all: torch.Tensor,
        generation_roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate final generated patches with warping and blending"""
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
        roi_mask_flat = generation_roi_mask.view(B * num_patches, 1, 1, 1)
        predicted_img_patches = torch.where(roi_mask_flat, roi_pred, non_roi_pred)
        
        # Alpha blending
        alpha_all = torch.where(
            generation_roi_mask,
            torch.ones_like(alpha_all),
            alpha_all
        )
        alpha_img = alpha_all.view(B * num_patches, 1, 1, 1)
        blended = alpha_img * predicted_img_patches + (1.0 - alpha_img) * curr_patches_img
        
        # Reshape back to patch format
        return blended.view(B, num_patches, -1)
        

class PointCloudGenerationModule(nn.Module):
    
    def __init__(self, 
                 prismatic_hidden_dim: int = 4096,
                 trans_dim: int = 1024,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 8,
                 group_size: int = 32,
                 num_groups: int = 128,
                 loss: str = 'cdl2',
                 use_geometric_prior: bool = True):
        super().__init__()
        
        self.prismatic_hidden_dim = prismatic_hidden_dim
        self.trans_dim = trans_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.group_size = group_size
        self.num_groups = num_groups
        self.loss = loss
        self.use_geometric_prior = use_geometric_prior
        
        self.feature_projector = nn.Linear(prismatic_hidden_dim, trans_dim)
        
        self.seq_to_patch = nn.Linear(trans_dim, num_groups * trans_dim)

        if use_geometric_prior:
            self.fps_sampler = FPSSampling(num_groups)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_groups, trans_dim))
        trunc_normal_(self.pos_embed, std=.02)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=trans_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                drop_path=0.1
            ) for _ in range(decoder_depth)
        ])
        
        self.future_predictor = nn.Sequential(
            nn.Conv1d(trans_dim, trans_dim, 1),
            nn.BatchNorm1d(trans_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(trans_dim, 3 * group_size, 1)
        )
    
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self,
                last_hidden: torch.Tensor,
                current_pointcloud: Optional[torch.Tensor] = None,
                vis: bool = False) -> torch.Tensor:
        B, seq_len, hidden_dim = last_hidden.shape

        projected_features = self.feature_projector(last_hidden)  # (B, seq_len, trans_dim)

        aggregated_features = projected_features.mean(dim=1)  # (B, trans_dim)
        patch_features = self.seq_to_patch(aggregated_features)  # (B, num_groups * trans_dim)
        patch_features = patch_features.reshape(B, self.num_groups, self.trans_dim)  # (B, G, trans_dim)

        patch_centers = None
        if self.use_geometric_prior and current_pointcloud is not None:
            patch_centers = self.fps_sampler(current_pointcloud)  # (B, G, 3)

        pos_features = self.pos_embed.expand(B, -1, -1)  # (B, G, trans_dim)

        future_features = patch_features
        for block in self.decoder_blocks:
            future_features = block(future_features, pos_features)  # (B, G, trans_dim)

        future_deltas = self.future_predictor(future_features.transpose(1, 2))  # (B, 3*M, G)
        future_deltas = future_deltas.transpose(1, 2).reshape(B * self.num_groups, self.group_size, 3)  # (B*G, M, 3)

        if patch_centers is not None:
            centers_expanded = patch_centers.reshape(B * self.num_groups, 1, 3).expand(-1, self.group_size, -1)
            pred_future_points = future_deltas + centers_expanded  # (B*G, M, 3)
        else:
            pred_future_points = future_deltas  # (B*G, M, 3)

        pred_future_pointcloud = pred_future_points.reshape(B, self.num_groups * self.group_size, 3)  # (B, G*M, 3)
        
        return {
            'pointcloud_coord_generation': pred_future_pointcloud,
        }


class TactileGenerationModule(nn.Module):
    def __init__(self,
                 token_size: int = 4096,
                 tactile_dim: int = 128,
                 decoder_layers: int = 2,
                 decoder_heads: int = 4):
        super().__init__()
        self.token_size = token_size
        self.tactile_dim = tactile_dim

        self.feature_projector = nn.Linear(token_size, token_size)

        # learnable query
        self.tactile_query = nn.Parameter(torch.zeros(1, 1, token_size))
        nn.init.normal_(self.tactile_query, std=0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_size,
            nhead=decoder_heads,
            dim_feedforward=token_size * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        self.output_head = nn.Linear(token_size, tactile_dim)

    def forward(self, llm_hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = llm_hidden_states.shape[0]

        # query expand
        query = self.tactile_query.expand(B, -1, -1)  # (B, 1, token_size)
        memory = self.feature_projector(llm_hidden_states)  # (B, seq_len, token_size)

        decoded = self.decoder(tgt=query, memory=memory)  # (B, 1, token_size)
        tactile_generation = self.output_head(decoded.squeeze(1))  # (B, tactile_dim)

        return {
            "tactile_generation": tactile_generation
        }


class MultimodalGenerationManager(nn.Module):
    def __init__(
        self,
        token_size: int = 4096,
        # Image generation parameters
        use_image_generation: bool = False,
        num_image_gen_queries: int = 64,
        image_decoder_layers: int = 3,
        image_decoder_heads: int = 8,
        image_patch_size: int = 42,
        use_roi: bool = True,
        roi_dilation_kernel_size: int = 3,
        # Point cloud generation parameters
        use_pointcloud_generation: bool = False,
        pointcloud_trans_dim: int = 1024,
        pointcloud_decoder_layers: int = 4,
        pointcloud_decoder_heads: int = 8,
        pointcloud_group_size: int = 16,
        pointcloud_num_groups: int = 64,
        # Tactile generation parameters
        use_tactile_generation: bool = False,
        tactile_dim: int = 128,
        tactile_decoder_layers: int = 2,
        tactile_decoder_heads: int = 4,
    ):
        super().__init__()
        
        self.use_image_generation = use_image_generation
        self.use_pointcloud_generation = use_pointcloud_generation
        self.use_tactile_generation = use_tactile_generation
        
        # Initialize generation modules
        if self.use_image_generation:
            self.image_gen_module = ImageGenerationModule(
                token_size=token_size,
                num_gen_queries=num_image_gen_queries,
                decoder_layers=image_decoder_layers,
                decoder_heads=image_decoder_heads,
                image_patch_size=image_patch_size,
                use_roi=use_roi,
                roi_dilation_kernel_size=roi_dilation_kernel_size,
            )
        
        if self.use_pointcloud_generation:
            self.pointcloud_gen_module = PointCloudGenerationModule(
                prismatic_hidden_dim=token_size,
                trans_dim=pointcloud_trans_dim,
                decoder_depth=pointcloud_decoder_layers,
                decoder_num_heads=pointcloud_decoder_heads,
                group_size=pointcloud_group_size,
                num_groups=pointcloud_num_groups,
                loss='cdl2',
                use_geometric_prior=True
            )
        
        if self.use_tactile_generation:
            self.tactile_gen_module = TactileGenerationModule(
                token_size=token_size,
                tactile_dim=tactile_dim,
                decoder_layers=tactile_decoder_layers,
                decoder_heads=tactile_decoder_heads,
            )
    
    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        current_image_features: Optional[torch.Tensor] = None,
        current_images_patches: Optional[torch.Tensor] = None,
        current_point_cloud: Optional[torch.Tensor] = None,
        roi_mask_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        generation_outputs = {}
        
        if self.use_image_generation:
            image_outputs = self.image_gen_module(
                llm_hidden_states=llm_hidden_states,
                current_image_features=current_image_features,
                current_images_patches=current_images_patches,
                roi_mask_2d=roi_mask_2d,
            )
            generation_outputs.update(image_outputs)
        
        if self.use_pointcloud_generation:
            pointcloud_outputs = self.pointcloud_gen_module(
                last_hidden=llm_hidden_states,
                current_pointcloud=current_point_cloud,
            )
            generation_outputs.update(pointcloud_outputs)
        
        if self.use_tactile_generation:
            tactile_outputs = self.tactile_gen_module(
                llm_hidden_states=llm_hidden_states
            )
            generation_outputs.update(tactile_outputs)
        
        return generation_outputs
    
    def get_module_keys(self) -> list:
        """Get all module keys for checkpoint saving/loading"""
        keys = []
        if self.use_image_generation:
            keys.append("image_gen_module")
        if self.use_pointcloud_generation:
            keys.append("pointcloud_gen_module")
        if self.use_tactile_generation:
            keys.append("tactile_gen_module")
        return keys