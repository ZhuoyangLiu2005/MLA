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
        recon_image: bool = False,
        recon_pointcloud: bool = True,
        num_image_recon_queries: int = 32,
        num_pointcloud_recon_queries: int = 32,
        recon_decoder_layers: int = 3,
        recon_decoder_heads: int = 8,
        image_patch_size: int = 42, 
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

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
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

        self.vision_tower_3d = PointViT(in_channels=3,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12,
                                        mlp_ratio=4.,
                                        qkv_bias=True,
                                        base_ckpt_path="/media/liuzhuoyang/new_vla/Any2Point/Any2Point_CLIP_Lang/ckpts/ViT-L-14.pt")
        self.projector_3d = MLPProjector(self.vision_tower_3d.embed_dim, token_size)

        # === Diffusion Components ===
        self.proprio_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)
        if self.use_diff:
            self.x_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)
            self.t_embedder = TimestepEmbedder(token_size)
            self.z_embedder = LabelEmbedder(in_size=token_size, hidden_size=token_size, dropout_prob=self.class_dropout_prob)
            self.final_layer = FinalLayer(token_size, action_dim)
            
        # === Reconstruction Components ===
        if self.use_reconstruction:
            self._setup_reconstruction_modules(token_size, recon_decoder_layers, recon_decoder_heads)

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_tower_2d", "vision_tower_3d","projector_2d", "projector_3d",
                                "llm_backbone", "proprio_embedder"]
        if self.use_diff:
            self.all_module_keys.extend(["x_embedder", "t_embedder", "final_layer",])
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
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=token_size,
                nhead=decoder_heads,
                dim_feedforward=token_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            self.image_recon_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
            self.image_recon_projector = nn.Linear(token_size, self.mm_hidden_size)
            self.image_num_patches = (672 // self.image_patch_size) ** 2  # 16 * 16 = 256
            self.image_mask_tokens = nn.Parameter(
                torch.zeros(1, self.image_num_patches, self.mm_hidden_size)
            )
            self.image_recon_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_image_recon_queries + self.image_num_patches, self.mm_hidden_size)
            )
            self.image_patch_predictor = nn.Sequential(
                nn.LayerNorm(self.mm_hidden_size),
                nn.Linear(self.mm_hidden_size, self.image_patch_size ** 2 * 3)  # RGB channels
            )

        # === PointCloud Reconstruction Components ===
        if self.recon_pointcloud:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=token_size,
                nhead=decoder_heads,
                dim_feedforward=token_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.pointcloud_recon_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
            self.pointcloud_recon_projector = nn.Linear(token_size, 768)  
            self.pointcloud_num_patches = 1024 
            self.pointcloud_mask_tokens = nn.Parameter(
                torch.zeros(1, self.pointcloud_num_patches, 768)
            )
            self.pointcloud_recon_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_pointcloud_recon_queries + self.pointcloud_num_patches, 768)
            )
            self.pointcloud_coord_predictor = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 3)  # XYZ coordinates
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
                "pointcloud_coord_predictor",
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
                self._init_pos_embed(self.image_recon_pos_embed)
                
            if self.recon_pointcloud:
                nn.init.normal_(self.pointcloud_recon_queries, std=0.02)
                nn.init.normal_(self.pointcloud_mask_tokens, std=0.02)
                self._init_pos_embed(self.pointcloud_recon_pos_embed)

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
    

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def get_fused_tokens(
        self, images, pointcloud
    ):
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
        assert N_pc == N_img # assert that the number of tokens are equal
        
        projected_fused_tokens = torch.cat(
            [projected_pointcloud_tokens, projected_image_tokens], 
            dim=1
        )

        return projected_fused_tokens, patch_indices, valid_mask
    
    def reconstruct_modalities(self, llm_hidden_states, batch_size):
        reconstruction_outputs = {}
        memory = llm_hidden_states  # [B, seq_len, token_size]
        
        if self.recon_image:
            # === Image Reconstruction ===
            image_queries = self.image_recon_queries.expand(batch_size, -1, -1)  # [B, num_queries, token_size]
            image_recon_features = self.image_recon_decoder(
                tgt=image_queries,
                memory=memory
            )  # [B, num_queries, token_size]
            image_recon_proj = self.image_recon_projector(image_recon_features)  # [B, num_queries, mm_hidden_size]
            image_mask_tokens = self.image_mask_tokens.expand(batch_size, -1, -1)  # [B, num_patches, mm_hidden_size]
            image_decoder_input = torch.cat([image_recon_proj, image_mask_tokens], dim=1)  # [B, num_queries + num_patches, mm_hidden_size]
            image_decoder_input = image_decoder_input + self.image_recon_pos_embed
            image_mask_features = image_decoder_input[:, self.num_image_recon_queries:, :]  # [B, num_patches, mm_hidden_size]
            image_patch_pred = self.image_patch_predictor(image_mask_features)  # [B, num_patches, patch_size^2 * 3]
            
            reconstruction_outputs['image_reconstruction'] = image_patch_pred

        if self.recon_pointcloud:
            # === PointCloud Reconstruction ===
            pc_queries = self.pointcloud_recon_queries.expand(batch_size, -1, -1)  # [B, num_queries, token_size]
            pc_recon_features = self.pointcloud_recon_decoder(
                tgt=pc_queries,
                memory=memory
            )  # [B, num_queries, token_size]
            pc_recon_proj = self.pointcloud_recon_projector(pc_recon_features)  # [B, num_queries, 768]
            pc_mask_tokens = self.pointcloud_mask_tokens.expand(batch_size, -1, -1)  # [B, num_patches, 768]
            pc_decoder_input = torch.cat([pc_recon_proj, pc_mask_tokens], dim=1)  # [B, num_queries + num_patches, 768]
            pc_decoder_input = pc_decoder_input + self.pointcloud_recon_pos_embed
            pc_mask_features = pc_decoder_input[:, self.num_pointcloud_recon_queries:, :]  # [B, num_patches, 768]
            pc_coord_pred = self.pointcloud_coord_predictor(pc_mask_features)  # [B, num_patches, 3]
            
            reconstruction_outputs['pointcloud_coord_reconstruction'] = pc_coord_pred

        return reconstruction_outputs
    
    def compute_reconstruction_losses(self, reconstruction_outputs, next_images=None, next_point_cloud=None):
        losses = {}
        total_loss = 0.0
        
        if self.recon_image and 'image_reconstruction' in reconstruction_outputs and next_images is not None:
            B, C, H, W = next_images.shape
            next_images_patches = self._images_to_patches_fixed(next_images)  # [B, num_patches, patch_size^2 * 3]
            
            image_pred = reconstruction_outputs['image_reconstruction']
            image_recon_loss = F.mse_loss(image_pred, next_images_patches)
            losses['image_reconstruction_loss'] = image_recon_loss
            total_loss += image_recon_loss

        def chamfer_distance(pred, gt):
            # pred/gt: [B, N, 3]
            dist = torch.cdist(pred, gt)  # [B, N, M]
            loss = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
            return loss

        if self.recon_pointcloud and next_point_cloud is not None:
            if 'pointcloud_coord_reconstruction' in reconstruction_outputs:
                # Extract coordinates from ground truth
                assert next_point_cloud.shape[2] == 3, "Point cloud must have 3 dimensions (XYZ)"

                pc_coord_pred = reconstruction_outputs['pointcloud_coord_reconstruction']
                pc_coord_loss = chamfer_distance(pc_coord_pred, next_point_cloud) 
                losses['pointcloud_coord_loss'] = pc_coord_loss
                total_loss += pc_coord_loss

        losses['total_reconstruction_loss'] = total_loss
        return losses
    
    def _images_to_patches_fixed(self, images):
        """Fixed version: Convert images to patches while preserving color information"""
        B, C, H, W = images.shape
        patch_size = self.image_patch_size
        
        # 确保输入是3通道RGB图像
        assert C == 3, f"Expected 3 channels (RGB), got {C}"
        assert H == W == 672, f"Expected 672x672 image, got {H}x{W}"
        
        # 使用unfold创建patches: [B, C, H, W] -> patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches shape: [B, C=3, num_patches_h=16, num_patches_w=16, patch_h=42, patch_w=42]
        
        # 重新排列以获得正确的patch顺序和形状
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        # patches shape: [B, C=3, num_patches=256, patch_h=42, patch_w=42]
        
        # 转置并重塑为最终形状: [B, num_patches, C * patch_h * patch_w]
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # patches shape: [B, num_patches=256, C=3, patch_h=42, patch_w=42]
        
        patches = patches.view(B, -1, C * patch_size * patch_size)
        # patches shape: [B, num_patches=256, C*patch_h*patch_w=5292]
        
        return patches
    
    def _patches_to_images_fixed(self, patches):
        """Fixed version: Convert patches back to images while preserving color information"""
        B, num_patches, patch_dim = patches.shape
        patch_size = self.image_patch_size
        
        expected_patch_dim = 3 * patch_size * patch_size
        assert patch_dim == expected_patch_dim, f"Expected patch_dim={expected_patch_dim}, got {patch_dim}"
        assert num_patches == 256, f"Expected 256 patches, got {num_patches}"
        
        # 重塑patches: [B, num_patches, C*patch_h*patch_w] -> [B, num_patches, C, patch_h, patch_w]
        patches = patches.view(B, num_patches, 3, patch_size, patch_size)
        # patches shape: [B, 256, 3, 42, 42]
        
        # 重新排列为网格形状: [B, num_patches_h, num_patches_w, C, patch_h, patch_w]
        num_patches_h = num_patches_w = int(num_patches ** 0.5)  # 16
        patches = patches.view(B, num_patches_h, num_patches_w, 3, patch_size, patch_size)
        
        # 重新组合成完整图像: [B, C, H, W]
        images = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        # images shape: [B, 3, 16, 42, 16, 42]
        images = images.view(B, 3, num_patches_h * patch_size, num_patches_w * patch_size)
        # images shape: [B, 3, 672, 672]
        
        return images
    
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
        gen_discret_action:  Optional[torch.LongTensor] = None,
        use_diff: Optional[bool] = None,
        # === Reconstruction Ground Truth ===
        next_images: Optional[torch.FloatTensor] = None,
        next_point_cloud: Optional[torch.FloatTensor] = None,
        **kwargs,
    ): 
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        if use_diff is not None:
            self.use_diff = use_diff
            
        # ......(保持原有的前向传播逻辑)

        fused_embeddings = multimodal_embeddings
        fused_attention_mask = multimodal_attention_mask
        fused_labels = multimodal_labels
        
        # Run LLM Forward --> returns CausalLMOutputWithPast!
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
            
            if next_images is not None and self.recon_image:
                reconstruction_losses = self.compute_reconstruction_losses(
                    reconstruction_outputs, 
                    next_images=next_images, 
                    next_point_cloud=next_point_cloud
                )
        
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
            if self.training and self.recon_image:
                # 使用修复后的可视化函数
                self.visualize_reconstruction_fixed(reconstruction_outputs, next_images, "/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vis/recon_vis_color_fixed")
                return output, noise_pred, reconstruction_outputs, reconstruction_losses
            else:
                return output, noise_pred
        
        # Return with reconstruction information
        if self.use_reconstruction and self.training and self.recon_image:
            # 使用修复后的可视化函数
            self.visualize_reconstruction_fixed(reconstruction_outputs, next_images, "/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vis/recon_vis_color_fixed")
            return output, reconstruction_outputs, reconstruction_losses
        else:
            return output # autoregressive
    
    def visualize_reconstruction_fixed(self, reconstruction_outputs, next_images, save_dir):
        """Fixed visualization function with proper color handling"""
        import os
        import matplotlib.pyplot as plt
        from torchvision.utils import save_image
        
        os.makedirs(save_dir, exist_ok=True)
        
        if 'image_reconstruction' in reconstruction_outputs and next_images is not None:
            # 将patch预测转换回图像
            patch_pred = reconstruction_outputs['image_reconstruction']  # [B, num_patches, patch_dim]
            reconstructed_images = self._patches_to_images_fixed(patch_pred)  # [B, 3, 672, 672]
            
            # 确保图像值在合理范围内 
            reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
            next_images = torch.clamp(next_images, 0, 1)
            
            batch_size = min(4, reconstructed_images.shape[0])  # 最多显示4个样本
            
            for i in range(batch_size):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # 原始图像
                target_img = next_images[i].cpu().detach().permute(1, 2, 0).numpy()
                axes[0, 0].imshow(target_img)
                axes[0, 0].set_title('Target Image')
                axes[0, 0].axis('off')
                
                # 重建图像
                recon_img = reconstructed_images[i].cpu().detach().permute(1, 2, 0).numpy()
                axes[0, 1].imshow(recon_img)
                axes[0, 1].set_title('Reconstructed Image')
                axes[0, 1].axis('off')
                
                # 差异图
                diff_img = abs(target_img - recon_img)
                axes[0, 2].imshow(diff_img)
                axes[0, 2].set_title('Absolute Difference')
                axes[0, 2].axis('off')
                
                # 显示各个颜色通道
                target_tensor = next_images[i].cpu().detach()
                recon_tensor = reconstructed_images[i].cpu().detach()
                
                axes[1, 0].imshow(target_tensor[0], cmap='Reds')
                axes[1, 0].set_title('Target Red Channel')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(recon_tensor[1], cmap='Greens')
                axes[1, 1].set_title('Reconstructed Green Channel')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(recon_tensor[2], cmap='Blues')
                axes[1, 2].set_title('Reconstructed Blue Channel')
                axes[1, 2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'reconstruction_comparison_{i}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                # 单独保存图像用于检查
                save_image(next_images[i], os.path.join(save_dir, f'target_{i}.png'))
                save_image(reconstructed_images[i], os.path.join(save_dir, f'reconstructed_{i}.png'))
                
            print(f"Color-fixed reconstruction visualization saved to {save_dir}")
            print(f"Target image shape: {next_images.shape}")
            print(f"Reconstructed image shape: {reconstructed_images.shape}")
            print(f"Target image range: [{next_images.min():.4f}, {next_images.max():.4f}]")
            print(f"Reconstructed image range: [{reconstructed_images.min():.4f}, {reconstructed_images.max():.4f}]")


# === 调试工具函数 ===

def debug_patch_conversion(images, patch_size=42):
    """调试patch转换过程，确保颜色信息正确保持"""
    print("=== Debug Patch Conversion ===")
    
    # 创建临时VLM实例用于测试patch转换
    class TempVLM:
        def __init__(self):
            self.image_patch_size = patch_size
            
        def _images_to_patches_fixed(self, images):
            B, C, H, W = images.shape
            patch_size = self.image_patch_size
            
            assert C == 3, f"Expected 3 channels (RGB), got {C}"
            assert H == W == 672, f"Expected 672x672 image, got {H}x{W}"
            
            patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
            patches = patches.permute(0, 2, 1, 3, 4).contiguous()
            patches = patches.view(B, -1, C * patch_size * patch_size)
            
            return patches
        
        def _patches_to_images_fixed(self, patches):
            B, num_patches, patch_dim = patches.shape
            patch_size = self.image_patch_size
            
            expected_patch_dim = 3 * patch_size * patch_size
            assert patch_dim == expected_patch_dim, f"Expected patch_dim={expected_patch_dim}, got {patch_dim}"
            assert num_patches == 256, f"Expected 256 patches, got {num_patches}"
            
            patches = patches.view(B, num_patches, 3, patch_size, patch_size)
            num_patches_h = num_patches_w = int(num_patches ** 0.5)
            patches = patches.view(B, num_patches_h, num_patches_w, 3, patch_size, patch_size)
            
            images = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
            images = images.view(B, 3, num_patches_h * patch_size, num_patches_w * patch_size)
            
            return images
    
    temp_vlm = TempVLM()
    
    print(f"Input images shape: {images.shape}")
    print(f"Input images range: [{images.min():.4f}, {images.max():.4f}]")
    
    # 转换为patches
    patches = temp_vlm._images_to_patches_fixed(images)
    print(f"Patches shape: {patches.shape}")
    print(f"Patches range: [{patches.min():.4f}, {patches.max():.4f}]")
    
    # 转换回图像
    reconstructed = temp_vlm._patches_to_images_fixed(patches)
    print(f"Reconstructed images shape: {reconstructed.shape}")
    print(f"Reconstructed images range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    
    # 检查是否完全一致
    diff = torch.abs(images - reconstructed)
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    
    if diff.max() < 1e-6:
        print("✅ Patch conversion is lossless!")
    else:
        print("❌ Patch conversion has errors!")
        
    return patches, reconstructed