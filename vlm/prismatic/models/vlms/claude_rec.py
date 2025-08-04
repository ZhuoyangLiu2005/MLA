import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from transformers.modeling_outputs import CausalLMOutputWithPast

class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        action_dim = 7,
        token_size = 4096,
        future_action_window_size=0,
        past_action_window_size=0,
        class_dropout_prob=0.0,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        use_diff = False,
        llm_vision_layers: int = 1,
        llm_action_layers: int = 1,
        # === Reconstruction Parameters ===
        use_reconstruction: bool = False,
        recon_image: bool = True,
        recon_pointcloud: bool = True,
        num_image_recon_queries: int = 32,
        num_pointcloud_recon_queries: int = 32,
        recon_decoder_layers: int = 3,
        recon_decoder_heads: int = 8,
        image_patch_size: int = 14,  # EVE patch size
        pointcloud_patch_size: int = 32,  # PointViT patch size
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )
        self.use_diff = use_diff
        self.llm_vision_layers = llm_vision_layers
        self.llm_action_layers = llm_action_layers
        
        # === Reconstruction Configuration ===
        self.use_reconstruction = use_reconstruction
        self.recon_image = recon_image and use_reconstruction
        self.recon_pointcloud = recon_pointcloud and use_reconstruction
        self.num_image_recon_queries = num_image_recon_queries
        self.num_pointcloud_recon_queries = num_pointcloud_recon_queries
        self.image_patch_size = image_patch_size
        self.pointcloud_patch_size = pointcloud_patch_size

        # === Generation Utilities ===
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

        self.norm_stats = norm_stats
        self.class_dropout_prob = class_dropout_prob
        self.future_action_window_size = future_action_window_size
        self.action_dim = action_dim
        
        self.mm_hidden_size = 1024
        self.vision_tower_2d = VisionTokenizer(input_size=self.mm_hidden_size, 
                                            vision_tower_name="/media/liuzhuoyang/new_vla/EVE/EVEv1/openai/eve-patch14-anypixel-672")
        self.projector_2d = MLP_GELU(self.mm_hidden_size, token_size, 2)

        # 3D Vision Tower
        self.vision_tower_3d = PointViT(in_channels=3,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12,
                                        mlp_ratio=4.,
                                        qkv_bias=True,
                                        base_ckpt_path="/media/liuzhuoyang/new_vla/Any2Point/Any2Point_CLIP_Lang/ckpts/ViT-L-14.pt")
        self.projector_3d = MLPProjector(self.vision_tower_3d.embed_dim, token_size)

        self.proprio_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)
        
        # === Diffusion Components ===
        if self.use_diff:
            self.x_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)
            self.t_embedder = TimestepEmbedder(token_size)
            self.z_embedder = LabelEmbedder(in_size=token_size, hidden_size=token_size, dropout_prob=self.class_dropout_prob)
            self.final_layer = FinalLayer(token_size, action_dim)

        # === Reconstruction Components ===
        if self.use_reconstruction:
            self._setup_reconstruction_modules(token_size, recon_decoder_layers, recon_decoder_heads)

        # Set Module Keys
        self.all_module_keys = ["vision_tower_2d", "vision_tower_3d","projector_2d", "projector_3d",
                                "llm_backbone", "proprio_embedder"]
        if self.use_diff:
            self.all_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
        if self.use_reconstruction:
            self.all_module_keys.extend(self._get_reconstruction_module_keys())
        
        self.trainable_module_keys = []

        self.initialize_weights()
        self.vision_tower_3d.initialize_weights()

    def _setup_reconstruction_modules(self, token_size, decoder_layers, decoder_heads):
        """Setup reconstruction-related modules"""
        # === Reconstruction Query Tokens ===
        if self.recon_image:
            self.image_recon_queries = nn.Parameter(
                torch.zeros(1, self.num_image_recon_queries, token_size)
            )
            
        if self.recon_pointcloud:
            self.pointcloud_recon_queries = nn.Parameter(
                torch.zeros(1, self.num_pointcloud_recon_queries, token_size)
            )

        # === Image Reconstruction Components ===
        if self.recon_image:
            # Image reconstruction decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=token_size,
                nhead=decoder_heads,
                dim_feedforward=token_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.image_recon_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
            
            # Image reconstruction heads
            self.image_recon_projector = nn.Linear(token_size, self.mm_hidden_size)
            
            # Calculate output dimensions for EVE (672x672 image, patch_size=14)
            self.image_num_patches = (672 // self.image_patch_size) ** 2  # 48*48 = 2304
            self.image_mask_tokens = nn.Parameter(
                torch.zeros(1, self.image_num_patches, self.mm_hidden_size)
            )
            
            # Position embeddings for image reconstruction
            self.image_recon_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_image_recon_queries + self.image_num_patches, self.mm_hidden_size)
            )
            
            # Final prediction head for image patches
            self.image_patch_predictor = nn.Sequential(
                nn.LayerNorm(self.mm_hidden_size),
                nn.Linear(self.mm_hidden_size, self.image_patch_size ** 2 * 3)  # RGB channels
            )

        # === PointCloud Reconstruction Components ===
        if self.recon_pointcloud:
            # PointCloud reconstruction decoder  
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=token_size,
                nhead=decoder_heads,
                dim_feedforward=token_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.pointcloud_recon_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
            
            # PointCloud reconstruction heads
            self.pointcloud_recon_projector = nn.Linear(token_size, 768)  # PointViT embed_dim
            
            # For pointcloud, we'll predict coordinates and features
            self.pointcloud_num_patches = 1024  # Typical number of point patches
            self.pointcloud_mask_tokens = nn.Parameter(
                torch.zeros(1, self.pointcloud_num_patches, 768)
            )
            
            # Position embeddings for pointcloud reconstruction
            self.pointcloud_recon_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_pointcloud_recon_queries + self.pointcloud_num_patches, 768)
            )
            
            # Final prediction heads for pointcloud
            self.pointcloud_coord_predictor = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 3)  # XYZ coordinates
            )
            self.pointcloud_feature_predictor = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 3)  # RGB features or other features
            )

    def _get_reconstruction_module_keys(self):
        """Get module keys for reconstruction components"""
        keys = []
        if self.recon_image:
            keys.extend([
                "image_recon_decoder", "image_recon_projector", "image_patch_predictor"
            ])
        if self.recon_pointcloud:
            keys.extend([
                "pointcloud_recon_decoder", "pointcloud_recon_projector", 
                "pointcloud_coord_predictor", "pointcloud_feature_predictor"
            ])
        return keys

    def get_vision_tower_2d(self):
        vision_tower_2d = getattr(self, 'vision_tower_2d', None)
        if type(vision_tower_2d) is list:
            vision_tower_2d = vision_tower_2d[0]
        return vision_tower_2d
    
    def encode_images(self, images):
        vision_tower_2d = self.get_vision_tower_2d()
        return vision_tower_2d(images, self.projector_2d)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize reconstruction components
        if self.use_reconstruction:
            if self.recon_image:
                nn.init.normal_(self.image_recon_queries, std=0.02)
                nn.init.normal_(self.image_mask_tokens, std=0.02)
                # Initialize position embeddings with sinusoidal patterns
                self._init_pos_embed(self.image_recon_pos_embed)
                
            if self.recon_pointcloud:
                nn.init.normal_(self.pointcloud_recon_queries, std=0.02)
                nn.init.normal_(self.pointcloud_mask_tokens, std=0.02)
                self._init_pos_embed(self.pointcloud_recon_pos_embed)

        # Initialize diffusion components
        if self.use_diff:
            nn.init.normal_(self.x_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.x_embedder.mlp.fc2.weight, std=0.02)
            nn.init.normal_(self.proprio_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.proprio_embedder.mlp.fc2.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            nn.init.constant_(self.final_layer.mlp.fc2.weight, 0)
            nn.init.constant_(self.final_layer.mlp.fc2.bias, 0)

    def _init_pos_embed(self, pos_embed):
        """Initialize position embeddings with sinusoidal encoding"""
        # Simple initialization - can be replaced with sinusoidal encoding
        nn.init.normal_(pos_embed, std=0.02)

    def freeze_backbones(self, stage: str) -> None:
        """
        Freeze backbones based on training stage
        """
        if stage == "align":
            pass
        elif stage in {"finetune", "vla-train"}:
            self.vision_tower_2d.requires_grad_(False)
            self.vision_tower_3d.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector_2d.requires_grad_(True)
            self.projector_3d.requires_grad_(True)

            # Add to trainable modules
            self.trainable_module_keys = ["llm_backbone","projector_2d","projector_3d", "proprio_embedder"]
            
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder","final_layer"])
                
            if self.use_reconstruction:
                self.trainable_module_keys.extend(self._get_reconstruction_module_keys())

            self.vision_backbone_requires_grad = False

    def get_fused_tokens(self, images, pointcloud):
        camera_params = {
            "K": torch.tensor([
                [-307.7174807,    0.0,         112.0],
                [   0.0,        -307.7174807,  112.0],
                [   0.0,           0.0,          1.0]
            ], dtype=torch.float32),
            "R": torch.tensor([
                [ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01],
                [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07],
                [-5.66244125e-07,  9.06307936e-01, -4.22617912e-01]
            ], dtype=torch.float32),
            "t": torch.tensor([1.34999919e+00, 3.71546562e-08, 1.57999933e+00], dtype=torch.float32)
        }
        
        # 2D tokens
        projected_image_tokens, patch_hw = self.encode_images(images)
        projected_image_tokens = torch.stack(projected_image_tokens, dim=0)
        
        # 3D tokens
        pointcloud_patch_embeddings, pointcloud_centers = self.vision_tower_3d(pointcloud)
        projected_pointcloud_tokens = self.projector_3d(pointcloud_patch_embeddings)
        
        # Project 3D to 2D for correspondence
        patch_indices, valid_mask = project_3d_to_2d_672_pyrep_compatible(
            pointcloud_centers, 
            camera_params['K'].to(pointcloud_centers.device),
            camera_params['R'].to(pointcloud_centers.device),
            camera_params['t'].to(pointcloud_centers.device),
            image_size_resize=(672, 672),
            vision_strides={'patch_stride': 14, 'conv_stride': 3}
        )
        
        N_pc = projected_pointcloud_tokens.shape[1]
        N_img = projected_image_tokens.shape[1]
        assert N_pc == N_img
        
        projected_fused_tokens = torch.cat(
            [projected_pointcloud_tokens, projected_image_tokens], 
            dim=1
        )

        return projected_fused_tokens, patch_indices, valid_mask

    def reconstruct_modalities(self, llm_hidden_states, batch_size):
        """
        Perform multimodal reconstruction using query tokens
        
        Args:
            llm_hidden_states: Hidden states from LLM backbone [B, seq_len, hidden_dim]
            batch_size: Batch size
            
        Returns:
            Dictionary containing reconstruction outputs and losses
        """
        reconstruction_outputs = {}
        
        # Extract memory for cross-attention (use all LLM hidden states as memory)
        memory = llm_hidden_states  # [B, seq_len, token_size]
        
        if self.recon_image:
            # === Image Reconstruction ===
            # Expand query tokens for batch
            image_queries = self.image_recon_queries.expand(batch_size, -1, -1)  # [B, num_queries, token_size]
            
            # Cross-attention with LLM hidden states
            image_recon_features = self.image_recon_decoder(
                tgt=image_queries,
                memory=memory
            )  # [B, num_queries, token_size]
            
            # Project to image feature space
            image_recon_proj = self.image_recon_projector(image_recon_features)  # [B, num_queries, mm_hidden_size]
            
            # Add mask tokens for reconstruction
            image_mask_tokens = self.image_mask_tokens.expand(batch_size, -1, -1)  # [B, num_patches, mm_hidden_size]
            
            # Concatenate query features and mask tokens
            image_decoder_input = torch.cat([image_recon_proj, image_mask_tokens], dim=1)  # [B, num_queries + num_patches, mm_hidden_size]
            
            # Add position embeddings
            image_decoder_input = image_decoder_input + self.image_recon_pos_embed
            
            # Predict image patches (only use mask token outputs)
            image_mask_features = image_decoder_input[:, self.num_image_recon_queries:, :]  # [B, num_patches, mm_hidden_size]
            image_patch_pred = self.image_patch_predictor(image_mask_features)  # [B, num_patches, patch_size^2 * 3]
            
            reconstruction_outputs['image_reconstruction'] = image_patch_pred

        if self.recon_pointcloud:
            # === PointCloud Reconstruction ===
            # Expand query tokens for batch  
            pc_queries = self.pointcloud_recon_queries.expand(batch_size, -1, -1)  # [B, num_queries, token_size]
            
            # Cross-attention with LLM hidden states
            pc_recon_features = self.pointcloud_recon_decoder(
                tgt=pc_queries,
                memory=memory
            )  # [B, num_queries, token_size]
            
            # Project to pointcloud feature space
            pc_recon_proj = self.pointcloud_recon_projector(pc_recon_features)  # [B, num_queries, 768]
            
            # Add mask tokens for reconstruction
            pc_mask_tokens = self.pointcloud_mask_tokens.expand(batch_size, -1, -1)  # [B, num_patches, 768]
            
            # Concatenate query features and mask tokens
            pc_decoder_input = torch.cat([pc_recon_proj, pc_mask_tokens], dim=1)  # [B, num_queries + num_patches, 768]
            
            # Add position embeddings
            pc_decoder_input = pc_decoder_input + self.pointcloud_recon_pos_embed
            
            # Predict pointcloud coordinates and features (only use mask token outputs)
            pc_mask_features = pc_decoder_input[:, self.num_pointcloud_recon_queries:, :]  # [B, num_patches, 768]
            pc_coord_pred = self.pointcloud_coord_predictor(pc_mask_features)  # [B, num_patches, 3]
            pc_feature_pred = self.pointcloud_feature_predictor(pc_mask_features)  # [B, num_patches, 3]
            
            reconstruction_outputs['pointcloud_coord_reconstruction'] = pc_coord_pred
            reconstruction_outputs['pointcloud_feature_reconstruction'] = pc_feature_pred

        return reconstruction_outputs

    def compute_reconstruction_losses(self, reconstruction_outputs, next_images=None, next_pointclouds=None):
        """
        Compute reconstruction losses
        
        Args:
            reconstruction_outputs: Outputs from reconstruct_modalities
            next_images: Ground truth next frame images [B, C, H, W]
            next_pointclouds: Ground truth next frame pointclouds [B, N, 6] (XYZ + RGB)
            
        Returns:
            Dictionary containing individual losses and total reconstruction loss
        """
        losses = {}
        total_loss = 0.0
        
        if self.recon_image and 'image_reconstruction' in reconstruction_outputs and next_images is not None:
            # Convert image to patches for comparison
            B, C, H, W = next_images.shape
            next_images_patches = self._images_to_patches(next_images)  # [B, num_patches, patch_size^2 * 3]
            
            image_pred = reconstruction_outputs['image_reconstruction']
            image_recon_loss = F.mse_loss(image_pred, next_images_patches)
            losses['image_reconstruction_loss'] = image_recon_loss
            total_loss += image_recon_loss

        if self.recon_pointcloud and next_pointclouds is not None:
            if 'pointcloud_coord_reconstruction' in reconstruction_outputs:
                # Extract coordinates from ground truth
                next_pc_coords = next_pointclouds[:, :, :3]  # [B, N, 3]
                
                # Subsample or pad to match prediction size
                next_pc_coords_resampled = self._resample_pointcloud(next_pc_coords, self.pointcloud_num_patches)
                
                pc_coord_pred = reconstruction_outputs['pointcloud_coord_reconstruction']
                pc_coord_loss = F.mse_loss(pc_coord_pred, next_pc_coords_resampled)
                losses['pointcloud_coord_loss'] = pc_coord_loss
                total_loss += pc_coord_loss
                
            if 'pointcloud_feature_reconstruction' in reconstruction_outputs and next_pointclouds.shape[-1] > 3:
                # Extract features from ground truth  
                next_pc_features = next_pointclouds[:, :, 3:6]  # [B, N, 3] RGB features
                next_pc_features_resampled = self._resample_pointcloud(next_pc_features, self.pointcloud_num_patches)
                
                pc_feature_pred = reconstruction_outputs['pointcloud_feature_reconstruction']
                pc_feature_loss = F.mse_loss(pc_feature_pred, next_pc_features_resampled)
                losses['pointcloud_feature_loss'] = pc_feature_loss
                total_loss += pc_feature_loss

        losses['total_reconstruction_loss'] = total_loss
        return losses

    def _images_to_patches(self, images):
        """Convert images to patches for reconstruction loss computation"""
        B, C, H, W = images.shape
        patch_size = self.image_patch_size
        
        # Reshape to patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * patch_size * patch_size)
        
        return patches

    def _resample_pointcloud(self, pointcloud, target_size):
        """Resample pointcloud to target size"""
        B, N, D = pointcloud.shape
        
        if N >= target_size:
            # Random sampling
            indices = torch.randperm(N)[:target_size]
            return pointcloud[:, indices, :]
        else:
            # Pad with zeros or repeat
            pad_size = target_size - N
            padding = torch.zeros(B, pad_size, D, device=pointcloud.device)
            return torch.cat([pointcloud, padding], dim=1)

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        z: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        gen_discret_action: Optional[torch.LongTensor] = None,
        use_diff: Optional[bool] = None,
        # === Reconstruction Ground Truth ===
        next_images: Optional[torch.FloatTensor] = None,
        next_pointclouds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """Run a forward pass through the VLM with optional reconstruction."""
        
        if use_diff is not None:
            self.use_diff = use_diff
            
        # Convert to bfloat16 if needed
        if proprio is not None:
            proprio = proprio.to(torch.bfloat16)
        if x is not None:
            x = x.to(torch.bfloat16)
        if t is not None:
            t = t.to(torch.bfloat16)
        if z is not None:
            z = z.to(torch.bfloat16)
        
        # Training/inference tags
        if self.training:
            tag_0, tag_1 = 2, 0
            tag_2 = 3
        else:
            tag_0, tag_1 = 29871, -1
            tag_2 = 0
        
        # Handle simple cases
        if input_ids.shape[1] == 1 and past_key_values is not None:
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or images is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle multimodal indices
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
        elif len(multimodal_indices) == 0:
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output, None
        
        # Get fused tokens
        point_cloud = point_cloud.to(self.device)
        projected_fused_embeddings, patch_indices, valid_mask = self.get_fused_tokens(images, point_cloud)
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        
        # Create initial embeddings with fused tokens
        z = torch.cat([
            input_embeddings[:, :1, :], 
            projected_fused_embeddings, 
            input_embeddings[:, 1:, :]
        ], dim=1)
        
        # Token layout tracking
        N_pc = projected_fused_embeddings.shape[1] // 2
        N_img = projected_fused_embeddings.shape[1] // 2
        bos_token_len = 1
        pc_tokens_start_idx = bos_token_len
        pc_tokens_end_idx = pc_tokens_start_idx + N_pc
        img_tokens_start_idx = pc_tokens_end_idx
        img_tokens_end_idx = img_tokens_start_idx + N_img

        # Process proprio and diffusion embeddings
        proprio = self.proprio_embedder(proprio)
        if self.use_diff:
            z = self.z_embedder(z, self.training)
            x = self.x_embedder(x)
            t = self.t_embedder(t).unsqueeze(1) if t is not None else None
        
        # Build multimodal embeddings with complex token layout
        multimodal_embeddings = []
        multimodal_attention_mask = []
        multimodal_labels = []
        last_true_indices = []
        
        if attention_mask is not None:
            projected_patch_attention_mask_fused = torch.full(
                (projected_fused_embeddings.shape[0], projected_fused_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        if labels is not None: 
            projected_patch_labels_fused = torch.full(
                (projected_fused_embeddings.shape[0], projected_fused_embeddings.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )

        for indice in multimodal_indices:
            if self.use_diff:
                last_true_indice = torch.where(input_ids[indice] == tag_0)[tag_1][-1].item() + projected_fused_embeddings.shape[1]
                last_true_indices.append(last_true_indice)
                embed = torch.cat([
                    z[indice, :last_true_indice, :],
                    proprio[indice],
                    t[indice] if t is not None else torch.zeros_like(proprio[indice]),
                    x[indice],
                    z[indice, last_true_indice:, :],
                ], dim=0).unsqueeze(0)
                multimodal_embeddings.append(embed)
            else:
                multimodal_embeddings.append(z[indice].unsqueeze(0))
                
            if attention_mask is not None:
                if self.use_diff:
                    attn_mask = torch.cat([
                        attention_mask[indice, :1],
                        projected_patch_attention_mask_fused[indice],
                        attention_mask[indice, 1:last_true_indice - projected_fused_embeddings.shape[1]],
                        torch.ones((proprio.shape[1]), dtype=torch.bool).to(projected_patch_attention_mask_fused.device),
                        torch.ones((t.shape[1] if t is not None else 0), dtype=torch.bool).to(projected_patch_attention_mask_fused.device),
                        torch.ones((x.shape[1]), dtype=torch.bool).to(projected_patch_attention_mask_fused.device),
                        attention_mask[indice, last_true_indice - projected_fused_embeddings.shape[1]:],
                    ], dim=0).unsqueeze(0)
                else:
                    attn_mask = torch.cat(
                        [
                            attention_mask[indice, :1],
                            projected_patch_attention_mask_fused[indice],
                            attention_mask[indice, 1:],
                        ],
                        dim=0,
                    ).unsqueeze(0)
                multimodal_attention_mask.append(attn_mask)

            if labels is not None:
                if self.use_diff:
                    label = torch.cat([
                        labels[indice, :1],
                        projected_patch_labels_fused[indice],
                        labels[indice, 1:last_true_indice - projected_fused_embeddings.shape[1]],
                        torch.full((proprio.shape[1],), -100).to(projected_patch_labels_fused.device),
                        torch.full((t.shape[1] if t is not None else 0,), -100).to(projected_patch_labels_fused.device),
                        torch.full((x.shape[1],), -100).to(projected_patch_labels_fused.device),
                        labels[indice, last_true_indice - projected_fused_embeddings.shape[1]:],
                    ], dim=0).unsqueeze(0)
                else:
                    label = torch.cat(
                        [
                            labels[indice, :1],
                            projected_patch_labels_fused[indice],
                            labels[indice, 1:],
                        ],
                        dim=0,
                    ).unsqueeze(0)
                multimodal_labels.append(label)
                
        multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)
        multimodal_attention_mask = torch.cat(multimodal_attention_mask, dim=0) if len(multimodal_attention_mask) != 0 else None
        multimodal_labels = torch.cat(multimodal_labels, dim=0) if len(multimodal_labels) != 0 else None

        fused_embeddings = multimodal_embeddings
        fused_attention_mask = multimodal_attention_mask
        fused_labels = multimodal_labels
        
        # Run LLM Forward
        output: CausalLMOutputWithPast = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            pc_token_indices=(pc_tokens_start_idx, pc_tokens_end_idx),
            img_token_indices=(img_tokens_start_idx, img_tokens_end_idx),
            patch_correspondence_indices=patch_indices,
            correspondence_valid_mask=valid_mask,
            compute_token_contrastive_loss=True, 
        )
        
        # === Reconstruction Forward Pass ===
        reconstruction_outputs = {}
        reconstruction_losses = {}
        
        if self.use_reconstruction and self.training and output.hidden_states is not None:
            # Use the last layer hidden states for reconstruction
            llm_hidden_states = output.hidden_states[-1]  # [B, seq_len, hidden_dim]
            
            # Perform multimodal reconstruction
            reconstruction_outputs = self.reconstruct_modalities(
                llm_hidden_states, 
                batch_size=fused_embeddings.shape[0]
            )
            
            # Compute reconstruction losses if ground truth is provided
            if next_images is not None or next_pointclouds is not None:
                reconstruction_losses = self.compute_reconstruction_losses(
                    reconstruction_outputs, 
                    next_images=next_images, 
                    next_pointclouds=next_pointclouds
                )
                
                # Add reconstruction loss to the main loss
                if 'total_reconstruction_loss' in reconstruction_losses and output.loss is not None:
                    # Combine LM loss with reconstruction loss (weighted)
                    recon_weight = 0.1  # Hyperparameter for reconstruction loss weight
                    output.loss = output.loss + recon_weight * reconstruction_losses['total_reconstruction_loss']
        
        # === Diffusion Action Prediction ===
        if self.use_diff:
            last_hidden = output.hidden_states[-1]
            last_hidden = self.final_layer(last_hidden)
            
            # Compute action output
            noise_pred = []
            for i, indices in enumerate(last_true_indices):
                noise_start = int(indices) + 2
                noise_end = int(indices) + self.future_action_window_size + 3
                noise_pred.append(last_hidden[i, noise_start:noise_end, :].unsqueeze(0))
            
            noise_pred = torch.cat(noise_pred, dim=0)
            
            # Return with reconstruction information
            return output, noise_pred, reconstruction_outputs, reconstruction_losses
        
        # Return with reconstruction information
        if self.use_reconstruction:
            return output, reconstruction_outputs, reconstruction_losses
        else:
            return output