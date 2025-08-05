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
        # === New Point Cloud Reconstruction Parameters ===
        pc_recon_strategy: str = "hierarchical",  # "hierarchical", "region_aware", "adaptive_sampling"
        pc_adaptive_sampling: bool = True,
        pc_region_weights: Dict[str, float] = None,
        pc_loss_type: str = "hybrid",  # "chamfer", "emd", "hybrid", "focal_chamfer"
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
        
        # === New Point Cloud Reconstruction Parameters ===
        self.pc_recon_strategy = pc_recon_strategy
        self.pc_adaptive_sampling = pc_adaptive_sampling
        self.pc_loss_type = pc_loss_type
        self.pc_region_weights = pc_region_weights or {
            'manipulated_object': 3.0,
            'end_effector': 2.0, 
            'arm': 1.0,
            'base': 0.5
        }

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

        # Set Module Keys
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
        """Setup reconstruction-related modules with improved point cloud reconstruction"""
        # === Reconstruction Query Tokens ===
        if self.recon_image:
            self.image_recon_queries = nn.Parameter(
                torch.zeros(1, self.num_image_recon_queries, token_size)
            )
            
        if self.recon_pointcloud:
            # Hierarchical queries for different regions
            if self.pc_recon_strategy == "hierarchical":
                self.pc_base_queries = nn.Parameter(torch.zeros(1, 8, token_size))
                self.pc_arm_queries = nn.Parameter(torch.zeros(1, 12, token_size))
                self.pc_end_effector_queries = nn.Parameter(torch.zeros(1, 8, token_size))
                self.pc_object_queries = nn.Parameter(torch.zeros(1, 16, token_size))
                total_queries = 8 + 12 + 8 + 16  # 44 queries
            else:
                self.pointcloud_recon_queries = nn.Parameter(
                    torch.zeros(1, self.num_pointcloud_recon_queries, token_size)
                )
                total_queries = self.num_pointcloud_recon_queries

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
            self.image_num_patches = (672 // self.image_patch_size) ** 2
            self.image_mask_tokens = nn.Parameter(
                torch.zeros(1, self.image_num_patches, self.mm_hidden_size)
            )
            self.image_recon_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_image_recon_queries + self.image_num_patches, self.mm_hidden_size)
            )
            self.image_patch_predictor = nn.Sequential(
                nn.LayerNorm(self.mm_hidden_size),
                nn.Linear(self.mm_hidden_size, self.image_patch_size ** 2 * 3)
            )

        # === Improved PointCloud Reconstruction Components ===
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
            
            # Multi-scale reconstruction heads
            self.pointcloud_recon_projector = nn.Linear(token_size, 768)
            
            # Adaptive point generation based on importance
            if self.pc_adaptive_sampling:
                self.pc_importance_predictor = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            
            # Region-specific predictors
            if self.pc_recon_strategy == "hierarchical":
                self.pc_base_predictor = self._make_pc_predictor(768, 200)  # Base: 200 points
                self.pc_arm_predictor = self._make_pc_predictor(768, 300)   # Arm: 300 points  
                self.pc_end_effector_predictor = self._make_pc_predictor(768, 150)  # End effector: 150 points
                self.pc_object_predictor = self._make_pc_predictor(768, 374)  # Object: 374 points
            else:
                self.pointcloud_num_patches = 1024
                self.pointcloud_mask_tokens = nn.Parameter(
                    torch.zeros(1, self.pointcloud_num_patches, 768)
                )
                # Remove position embeddings for point clouds since they're orderless
                self.pointcloud_coord_predictor = self._make_pc_predictor(768, None)
            
    def _make_pc_predictor(self, input_dim, num_points=None):
        """Create point cloud predictor with optional point count specification"""
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 3)  # XYZ coordinates
        ]
        return nn.Sequential(*layers)
            
    def _get_reconstruction_module_keys(self):
        """Get module keys for reconstruction components"""
        keys = []
        if self.recon_image:
            keys.extend([
                "image_recon_decoder", "image_recon_projector", "image_patch_predictor"
            ])
        if self.recon_pointcloud:
            keys.extend(["pointcloud_recon_decoder", "pointcloud_recon_projector"])
            if self.pc_recon_strategy == "hierarchical":
                keys.extend([
                    "pc_base_predictor", "pc_arm_predictor", 
                    "pc_end_effector_predictor", "pc_object_predictor"
                ])
            else:
                keys.append("pointcloud_coord_predictor")
            if self.pc_adaptive_sampling:
                keys.append("pc_importance_predictor")
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
                if self.pc_recon_strategy == "hierarchical":
                    nn.init.normal_(self.pc_base_queries, std=0.02)
                    nn.init.normal_(self.pc_arm_queries, std=0.02) 
                    nn.init.normal_(self.pc_end_effector_queries, std=0.02)
                    nn.init.normal_(self.pc_object_queries, std=0.02)
                else:
                    nn.init.normal_(self.pointcloud_recon_queries, std=0.02)
                    nn.init.normal_(self.pointcloud_mask_tokens, std=0.02)

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
        nn.init.normal_(pos_embed, std=0.02)

    def _segment_point_cloud(self, point_cloud, current_point_cloud=None):
        """
        Segment point cloud into different regions based on spatial characteristics
        Args:
            point_cloud: [B, N, 3] target point cloud
            current_point_cloud: [B, N, 3] current frame point cloud for reference
        Returns:
            region_masks: dict with region masks
            region_points: dict with segmented points
        """
        B, N, _ = point_cloud.shape
        
        # Simple heuristic segmentation based on height and distance from center
        # You can replace this with more sophisticated segmentation
        
        # Compute center and statistics
        center = point_cloud.mean(dim=1, keepdim=True)  # [B, 1, 3]
        distances = torch.norm(point_cloud - center, dim=2)  # [B, N]
        heights = point_cloud[:, :, 2]  # Z coordinate
        
        region_masks = {}
        region_points = {}
        
        # Base: lowest points with small movement
        base_mask = (heights < heights.quantile(0.3, dim=1, keepdim=True)) & \
                   (distances < distances.quantile(0.4, dim=1, keepdim=True))
        
        # Object: highest points or points with significant movement
        if current_point_cloud is not None:
            movement = torch.norm(point_cloud - current_point_cloud, dim=2)
            object_mask = (heights > heights.quantile(0.7, dim=1, keepdim=True)) | \
                         (movement > movement.quantile(0.8, dim=1, keepdim=True))
        else:
            object_mask = heights > heights.quantile(0.8, dim=1, keepdim=True)
        
        # End effector: medium height, medium distance, high movement
        if current_point_cloud is not None:
            movement = torch.norm(point_cloud - current_point_cloud, dim=2)
            end_effector_mask = (heights > heights.quantile(0.4, dim=1, keepdim=True)) & \
                               (heights < heights.quantile(0.8, dim=1, keepdim=True)) & \
                               (movement > movement.quantile(0.6, dim=1, keepdim=True))
        else:
            end_effector_mask = (heights > heights.quantile(0.5, dim=1, keepdim=True)) & \
                               (heights < heights.quantile(0.8, dim=1, keepdim=True)) & \
                               (distances > distances.quantile(0.6, dim=1, keepdim=True))
        
        # Arm: remaining points
        arm_mask = ~(base_mask | object_mask | end_effector_mask)
        
        region_masks = {
            'base': base_mask,
            'arm': arm_mask, 
            'end_effector': end_effector_mask,
            'manipulated_object': object_mask
        }
        
        # Extract points for each region
        for region, mask in region_masks.items():
            region_points[region] = point_cloud[mask.unsqueeze(-1).expand(-1, -1, 3)].view(B, -1, 3)
            
        return region_masks, region_points

    def reconstruct_modalities(self, llm_hidden_states, batch_size):
        reconstruction_outputs = {}
        memory = llm_hidden_states  # [B, seq_len, token_size]
        
        if self.recon_image:
            # === Image Reconstruction (unchanged) ===
            image_queries = self.image_recon_queries.expand(batch_size, -1, -1)
            image_recon_features = self.image_recon_decoder(
                tgt=image_queries,
                memory=memory
            )
            image_recon_proj = self.image_recon_projector(image_recon_features)
            image_mask_tokens = self.image_mask_tokens.expand(batch_size, -1, -1)
            image_decoder_input = torch.cat([image_recon_proj, image_mask_tokens], dim=1)
            image_decoder_input = image_decoder_input + self.image_recon_pos_embed
            image_mask_features = image_decoder_input[:, self.num_image_recon_queries:, :]
            image_patch_pred = self.image_patch_predictor(image_mask_features)
            
            reconstruction_outputs['image_reconstruction'] = image_patch_pred

        if self.recon_pointcloud:
            # === Improved PointCloud Reconstruction ===
            if self.pc_recon_strategy == "hierarchical":
                # Hierarchical reconstruction with region-specific queries
                all_queries = torch.cat([
                    self.pc_base_queries.expand(batch_size, -1, -1),
                    self.pc_arm_queries.expand(batch_size, -1, -1),
                    self.pc_end_effector_queries.expand(batch_size, -1, -1),
                    self.pc_object_queries.expand(batch_size, -1, -1)
                ], dim=1)
                
                pc_recon_features = self.pointcloud_recon_decoder(
                    tgt=all_queries,
                    memory=memory
                )
                pc_recon_proj = self.pointcloud_recon_projector(pc_recon_features)
                
                # Split features by region
                base_features = pc_recon_proj[:, :8, :]
                arm_features = pc_recon_proj[:, 8:20, :]
                end_effector_features = pc_recon_proj[:, 20:28, :]
                object_features = pc_recon_proj[:, 28:44, :]
                
                # Generate points for each region
                base_points = self.pc_base_predictor(base_features)  # [B, 8, 768] -> [B, 200, 3]
                arm_points = self.pc_arm_predictor(arm_features)    # [B, 12, 768] -> [B, 300, 3]
                end_effector_points = self.pc_end_effector_predictor(end_effector_features)  # [B, 8, 768] -> [B, 150, 3]
                object_points = self.pc_object_predictor(object_features)  # [B, 16, 768] -> [B, 374, 3]
                
                # Expand features to match target point counts
                base_points = base_features.unsqueeze(2).expand(-1, -1, 25, -1).reshape(batch_size, 200, 768)
                base_points = self.pc_base_predictor(base_points)
                
                arm_points = arm_features.unsqueeze(2).expand(-1, -1, 25, -1).reshape(batch_size, 300, 768)
                arm_points = self.pc_arm_predictor(arm_points)
                
                end_effector_points = end_effector_features.unsqueeze(2).expand(-1, -1, 19, -1).reshape(batch_size, 152, 768)[:, :150, :]
                end_effector_points = self.pc_end_effector_predictor(end_effector_points)
                
                object_points = object_features.unsqueeze(2).expand(-1, -1, 24, -1).reshape(batch_size, 384, 768)[:, :374, :]
                object_points = self.pc_object_predictor(object_points)
                
                reconstruction_outputs.update({
                    'pc_base_reconstruction': base_points,
                    'pc_arm_reconstruction': arm_points,
                    'pc_end_effector_reconstruction': end_effector_points,
                    'pc_object_reconstruction': object_points
                })
                
                # Combine all regions
                full_pc_reconstruction = torch.cat([
                    base_points, arm_points, end_effector_points, object_points
                ], dim=1)
                reconstruction_outputs['pointcloud_coord_reconstruction'] = full_pc_reconstruction
                
            else:
                # Standard reconstruction
                pc_queries = self.pointcloud_recon_queries.expand(batch_size, -1, -1)
                pc_recon_features = self.pointcloud_recon_decoder(
                    tgt=pc_queries,
                    memory=memory
                )
                pc_recon_proj = self.pointcloud_recon_projector(pc_recon_features)
                
                # No position embeddings for point clouds - they're orderless
                pc_mask_tokens = self.pointcloud_mask_tokens.expand(batch_size, -1, -1)
                pc_decoder_input = torch.cat([pc_recon_proj, pc_mask_tokens], dim=1)
                
                # Generate importance weights if adaptive sampling is enabled
                if self.pc_adaptive_sampling:
                    importance_weights = self.pc_importance_predictor(pc_decoder_input)
                    reconstruction_outputs['pc_importance_weights'] = importance_weights
                
                pc_mask_features = pc_decoder_input[:, self.num_pointcloud_recon_queries:, :]
                pc_coord_pred = self.pointcloud_coord_predictor(pc_mask_features)
                
                reconstruction_outputs['pointcloud_coord_reconstruction'] = pc_coord_pred

        return reconstruction_outputs
    
    def compute_reconstruction_losses(self, reconstruction_outputs, next_images=None, next_point_cloud=None, current_point_cloud=None):
        losses = {}
        total_loss = 0.0
        
        if self.recon_image and 'image_reconstruction' in reconstruction_outputs and next_images is not None:
            B, C, H, W = next_images.shape
            next_images_patches = self._images_to_patches(next_images)
            
            image_pred = reconstruction_outputs['image_reconstruction']
            image_recon_loss = F.mse_loss(image_pred, next_images_patches)
            losses['image_reconstruction_loss'] = image_recon_loss
            total_loss += image_recon_loss

        if self.recon_pointcloud and next_point_cloud is not None:
            if self.pc_recon_strategy == "hierarchical":
                # Segment ground truth point cloud
                region_masks, region_points = self._segment_point_cloud(
                    next_point_cloud, current_point_cloud
                )
                
                region_losses = {}
                for region in ['base', 'arm', 'end_effector', 'manipulated_object']:
                    if f'pc_{region}_reconstruction' in reconstruction_outputs:
                        pred_points = reconstruction_outputs[f'pc_{region}_reconstruction']
                        gt_points = region_points[region]
                        
                        # Compute weighted loss based on region importance
                        if self.pc_loss_type == "hybrid":
                            chamfer_loss = self._compute_chamfer_distance(pred_points, gt_points)
                            # Add surface normal consistency for objects
                            if region == 'manipulated_object':
                                normal_loss = self._compute_normal_consistency_loss(pred_points, gt_points)
                                region_loss = chamfer_loss + 0.1 * normal_loss
                            else:
                                region_loss = chamfer_loss
                        else:
                            region_loss = self._compute_chamfer_distance(pred_points, gt_points)
                        
                        weighted_loss = region_loss * self.pc_region_weights[region]
                        region_losses[f'{region}_loss'] = weighted_loss
                        total_loss += weighted_loss
                
                losses.update(region_losses)
                
            else:
                # Standard point cloud reconstruction
                pc_coord_pred = reconstruction_outputs['pointcloud_coord_reconstruction']
                
                if self.pc_loss_type == "focal_chamfer":
                    pc_loss = self._compute_focal_chamfer_loss(pc_coord_pred, next_point_cloud, current_point_cloud)
                elif self.pc_loss_type == "hybrid":
                    chamfer_loss = self._compute_chamfer_distance(pc_coord_pred, next_point_cloud)
                    emd_loss = self._compute_earth_movers_distance(pc_coord_pred, next_point_cloud)
                    pc_loss = 0.7 * chamfer_loss + 0.3 * emd_loss
                else:
                    pc_loss = self._compute_chamfer_distance(pc_coord_pred, next_point_cloud)
                
                losses['pointcloud_coord_loss'] = pc_loss
                total_loss += pc_loss

        losses['total_reconstruction_loss'] = total_loss
        return losses
    
    def _compute_chamfer_distance(self, pred, gt):
        """Compute Chamfer Distance between predicted and ground truth point clouds"""
        # pred/gt: [B, N, 3]
        dist_pred_to_gt = torch.cdist(pred, gt).min(dim=2)[0].mean(dim=1)  # [B]
        dist_gt_to_pred = torch.cdist(gt, pred).min(dim=2)[0].mean(dim=1)  # [B]
        return (dist_pred_to_gt + dist_gt_to_pred).mean()
    
    def _compute_earth_movers_distance(self, pred, gt):
        """Approximate Earth Mover's Distance using Sinkhorn algorithm"""
        # Simplified EMD approximation - you might want to use a more sophisticated implementation
        B, N, _ = pred.shape
        M = gt.shape[1]
        
        # Compute cost matrix
        cost_matrix = torch.cdist(pred, gt)  # [B, N, M]
        
        # Uniform distributions
        a = torch.ones(B, N, device=pred.device) / N
        b = torch.ones(B, M, device=pred.device) / M
        
        # Simplified Sinkhorn iterations (you can use proper implementation)
        for _ in range(10):
            a = a / (cost_matrix * b.unsqueeze(1)).sum(dim=2)
            b = b / (cost_matrix * a.unsqueeze(2)).sum(dim=1)
        
        transport_plan = a.unsqueeze(2) * b.unsqueeze(1) * torch.exp(-cost_matrix)
        emd = (transport_plan * cost_matrix).sum(dim=[1, 2]).mean()
        
        return emd
    
    def _compute_focal_chamfer_loss(self, pred, gt, current_pc=None):
        """
        Focal Chamfer Loss that emphasizes harder-to-reconstruct regions
        """
        # Compute standard chamfer distance for each point
        dist_pred_to_gt = torch.cdist(pred, gt)  # [B, N_pred, N_gt]
        dist_gt_to_pred = torch.cdist(gt, pred)  # [B, N_gt, N_pred]
        
        min_dist_pred_to_gt, _ = dist_pred_to_gt.min(dim=2)  # [B, N_pred]
        min_dist_gt_to_pred, _ = dist_gt_to_pred.min(dim=2)  # [B, N_gt]
        
        # Compute focal weights based on prediction difficulty
        # Points that are harder to predict get higher weights
        if current_pc is not None:
            # Points with more movement get higher weight
            movement = torch.norm(gt - current_pc, dim=2)  # [B, N_gt]
            movement_weights = 1 + 2 * (movement / (movement.max(dim=1, keepdim=True)[0] + 1e-8))
        else:
            movement_weights = torch.ones_like(min_dist_gt_to_pred)
        
        # Apply focal loss weighting
        alpha = 2.0  # focusing parameter
        focal_weight_pred = (1 + min_dist_pred_to_gt) ** alpha
        focal_weight_gt = (1 + min_dist_gt_to_pred) ** alpha
        
        # Combine with movement weights
        weighted_dist_pred = (focal_weight_pred * min_dist_pred_to_gt).mean(dim=1)
        weighted_dist_gt = (focal_weight_gt * movement_weights * min_dist_gt_to_pred).mean(dim=1)
        
        return (weighted_dist_pred + weighted_dist_gt).mean()
    
    def _compute_normal_consistency_loss(self, pred, gt):
        """
        Compute surface normal consistency loss for better surface reconstruction
        """
        def compute_normals(points):
            # Simple normal estimation using nearest neighbors
            # points: [B, N, 3]
            B, N, _ = points.shape
            
            # Find k nearest neighbors (k=8)
            k = min(8, N-1)
            distances = torch.cdist(points, points)  # [B, N, N]
            _, knn_idx = distances.topk(k+1, dim=2, largest=False)  # [B, N, k+1]
            knn_idx = knn_idx[:, :, 1:]  # Remove self, [B, N, k]
            
            # Get neighbor points
            batch_idx = torch.arange(B).view(B, 1, 1).expand(-1, N, k)
            point_idx = torch.arange(N).view(1, N, 1).expand(B, -1, k)
            neighbors = points[batch_idx, knn_idx]  # [B, N, k, 3]
            
            # Compute local normal using PCA
            centered = neighbors - points.unsqueeze(2)  # [B, N, k, 3]
            cov = torch.matmul(centered.transpose(-1, -2), centered)  # [B, N, 3, 3]
            
            # Get normal as eigenvector with smallest eigenvalue
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(cov)
                normals = eigenvecs[:, :, :, 0]  # Smallest eigenvalue eigenvector
            except:
                # Fallback to simple normal estimation
                normals = torch.cross(centered[:, :, 0], centered[:, :, 1], dim=-1)
                normals = F.normalize(normals, dim=-1)
            
            return normals
        
        pred_normals = compute_normals(pred)
        gt_normals = compute_normals(gt)
        
        # Find correspondences and compute normal consistency
        distances = torch.cdist(pred, gt)
        _, closest_gt_idx = distances.min(dim=2)  # [B, N_pred]
        
        # Get corresponding ground truth normals
        batch_idx = torch.arange(pred.shape[0]).view(-1, 1).expand(-1, pred.shape[1])
        corresponding_gt_normals = gt_normals[batch_idx, closest_gt_idx]
        
        # Compute cosine similarity between normals
        normal_similarity = F.cosine_similarity(pred_normals, corresponding_gt_normals, dim=-1)
        normal_loss = (1 - normal_similarity.abs()).mean()  # Penalize dissimilar normals
        
        return normal_loss
    
    def _images_to_patches(self, images):
        """Convert images to patches for reconstruction loss computation"""
        B, C, H, W = images.shape
        patch_size = self.image_patch_size
        
        # Reshape to patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * patch_size * patch_size)
        
        return patches
    
    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

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
            
        # ... (multimodal processing code - same as before)
        
        # Get fused tokens
        projected_fused_tokens, patch_indices, valid_mask = self.get_fused_tokens(images, point_cloud)
        
        # ... (rest of multimodal processing - same as before)
        
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
        
        # === Enhanced Reconstruction Forward Pass ===
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
            
            if next_images is not None or next_point_cloud is not None:
                reconstruction_losses = self.compute_reconstruction_losses(
                    reconstruction_outputs, 
                    next_images=next_images, 
                    next_point_cloud=next_point_cloud,
                    current_point_cloud=point_cloud  # Pass current frame for movement analysis
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
            
            if self.training:
                # Enhanced visualization for debugging
                if reconstruction_outputs:
                    self._visualize_enhanced_reconstruction(
                        reconstruction_outputs, next_images, next_point_cloud, point_cloud,
                        "/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vis/enhanced_recon_vis"
                    )
                return output, noise_pred, reconstruction_outputs, reconstruction_losses
            else:
                return output, noise_pred
        
        if self.use_reconstruction and self.training:
            return output, reconstruction_outputs, reconstruction_losses
        else:
            return output # autoregressive
    
    def _visualize_enhanced_reconstruction(self, reconstruction_outputs, next_images, next_point_cloud, current_point_cloud, save_dir):
        """Enhanced visualization for debugging reconstruction quality"""
        import os
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        os.makedirs(save_dir, exist_ok=True)
        
        if next_point_cloud is not None and 'pointcloud_coord_reconstruction' in reconstruction_outputs:
            # Visualize point cloud reconstruction
            pred_pc = reconstruction_outputs['pointcloud_coord_reconstruction'][0].detach().cpu().numpy()
            gt_pc = next_point_cloud[0].detach().cpu().numpy()
            
            fig = plt.figure(figsize=(15, 5))
            
            # Current frame
            if current_point_cloud is not None:
                ax1 = fig.add_subplot(131, projection='3d')
                current_pc = current_point_cloud[0].detach().cpu().numpy()
                ax1.scatter(current_pc[:, 0], current_pc[:, 1], current_pc[:, 2], 
                           c='blue', s=1, alpha=0.6)
                ax1.set_title('Current Frame')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
            
            # Ground truth next frame
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], 
                       c='green', s=1, alpha=0.6)
            ax2.set_title('Ground Truth Next Frame')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # Predicted next frame
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.scatter(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2], 
                       c='red', s=1, alpha=0.6)
            ax3.set_title('Predicted Next Frame')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'pointcloud_reconstruction.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # If hierarchical reconstruction, visualize regions separately
            if self.pc_recon_strategy == "hierarchical":
                region_colors = {
                    'pc_base_reconstruction': 'brown',
                    'pc_arm_reconstruction': 'blue', 
                    'pc_end_effector_reconstruction': 'orange',
                    'pc_object_reconstruction': 'red'
                }
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
                axes = axes.flatten()
                
                for i, (region, color) in enumerate(region_colors.items()):
                    if region in reconstruction_outputs:
                        pred_region = reconstruction_outputs[region][0].detach().cpu().numpy()
                        axes[i].scatter(pred_region[:, 0], pred_region[:, 1], pred_region[:, 2],
                                       c=color, s=2, alpha=0.7)
                        axes[i].set_title(f'Predicted {region.replace("pc_", "").replace("_reconstruction", "")}')
                        axes[i].set_xlabel('X')
                        axes[i].set_ylabel('Y')
                        axes[i].set_zlabel('Z')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'hierarchical_reconstruction.png'), dpi=150, bbox_inches='tight')
                plt.close()
        
        # Add movement analysis visualization
        if current_point_cloud is not None and next_point_cloud is not None:
            current_pc = current_point_cloud[0].detach().cpu().numpy()
            next_pc = next_point_cloud[0].detach().cpu().numpy()
            
            # Compute movement for each point
            movement = np.linalg.norm(next_pc - current_pc, axis=1)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color points by movement magnitude
            scatter = ax.scatter(current_pc[:, 0], current_pc[:, 1], current_pc[:, 2], 
                               c=movement, cmap='viridis', s=2, alpha=0.7)
            
            plt.colorbar(scatter, ax=ax, label='Movement Magnitude')
            ax.set_title('Point Movement Analysis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            plt.savefig(os.path.join(save_dir, 'movement_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()