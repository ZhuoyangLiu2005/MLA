import torch
import torch.nn as nn
import torch.nn.functional as F

def project_3d_to_2d_672_rlbench(
    xyz_3d,                     
    K, R, t,                    
    image_size_orig=(224, 224), 
    image_size_resize=(672, 672), 
    vision_strides={"patch_stride": 14, "conv_stride": 2},
):
    
    scale_x = image_size_resize[1] / image_size_orig[1]  # 672/224 = 3
    scale_y = image_size_resize[0] / image_size_orig[0]  # 672/224 = 3
    
    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_x 
    K_scaled[1, 1] *= scale_y  
    K_scaled[0, 2] *= scale_x  
    K_scaled[1, 2] *= scale_y  
    
    R_world_to_cam = R.T
    t_world_to_cam = -R_world_to_cam @ t
    
    xyz_cam = xyz_3d @ R_world_to_cam.T + t_world_to_cam
    
    uvw = xyz_cam @ K_scaled.T
    z = uvw[..., 2:]
    xy = uvw[..., :2] / (z + 1e-6)
    
    total_stride = vision_strides["patch_stride"] * vision_strides["conv_stride"]  # 14 * 3 = 42
    
    row = (xy[..., 1] / total_stride).floor().long()
    col = (xy[..., 0] / total_stride).floor().long()
    
    patch_h = image_size_resize[0] // total_stride  # 672 // 42 = 16
    patch_w = image_size_resize[1] // total_stride  # 672 // 42 = 16
    
    valid = (z.squeeze(-1) > 0) & \
            (xy[..., 0] >= 0) & (xy[..., 0] < image_size_resize[1]) & \
            (xy[..., 1] >= 0) & (xy[..., 1] < image_size_resize[0])

    row = torch.clamp(row, 0, patch_h - 1)
    col = torch.clamp(col, 0, patch_w - 1)
    patch_idx = torch.stack([row, col], dim=-1)
    
    return patch_idx, valid

def project_3d_to_2d_672_metaworld(
    xyz_3d,                      
    K, R, t,                     
    transform_key,               
    image_size_orig=(224, 224),
    image_size_resize=(672, 672),
    vision_strides={"patch_stride": 14, "conv_stride": 3},
):
    PC_TRANSFORM_TENSORS = {
        "identity": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32
        ),
        "corner": torch.tensor(
            [
                [-0.66173422, -0.48809537, 0.56909642],
                [-0.31361979, 0.86966611, 0.38121317],
                [0.68099225, -0.0737819, 0.7285642],
            ], dtype=torch.float32
        ),
        "corner2": torch.tensor(
            [
                [0.56914086, -0.56424844, 0.59808225],
                [0.23069754, 0.80774597, 0.54251738],
                [-0.78921311, -0.17079271, 0.58989196],
            ], dtype=torch.float32
        ),
    }

    if transform_key not in PC_TRANSFORM_TENSORS:
        raise ValueError(f"Unknown transform_key: '{transform_key}'. Available keys: {list(PC_TRANSFORM_TENSORS.keys())}")
    
    pc_transform_matrix = PC_TRANSFORM_TENSORS[transform_key].to(xyz_3d.device)
    xyz_3d_mujoco_frame = xyz_3d @ pc_transform_matrix
    
    scale_x = image_size_resize[1] / image_size_orig[1]
    scale_y = image_size_resize[0] / image_size_orig[0]
    
    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    
    # R' = R.T
    R_world_to_cam = R.T
    # t' = -R' @ t
    t_world_to_cam = -R_world_to_cam @ t
    
    xyz_cam = xyz_3d_mujoco_frame @ R_world_to_cam + t_world_to_cam
    
    uvw = xyz_cam @ K_scaled.T
    z = uvw[..., 2:]
    xy = uvw[..., :2] / (z + 1e-6)
    
    total_stride = vision_strides["patch_stride"] * vision_strides["conv_stride"]
    
    row = (xy[..., 1] / total_stride).floor().long()
    col = (xy[..., 0] / total_stride).floor().long()
    
    patch_h = image_size_resize[0] // total_stride
    patch_w = image_size_resize[1] // total_stride
    
    valid = (z.squeeze(-1) > 0) & \
            (xy[..., 0] >= 0) & (xy[..., 0] < image_size_resize[1]) & \
            (xy[..., 1] >= 0) & (xy[..., 1] < image_size_resize[0])

    row = torch.clamp(row, 0, patch_h - 1)
    col = torch.clamp(col, 0, patch_w - 1)
    patch_idx = torch.stack([row, col], dim=-1)
    
    return patch_idx, valid

def project_3d_to_2d_672_franka_right(
    xyz_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_size_orig: tuple = (480, 640),
    image_size_resize: tuple = (672, 672),
    vision_strides: dict = {"patch_stride": 14, "conv_stride": 2},
):
    
    scale_x = image_size_resize[1] / image_size_orig[1]  # 672/224 = 3
    scale_y = image_size_resize[0] / image_size_orig[0]  # 672/224 = 3
    
    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    
    R_world_to_cam = R.T
    t_world_to_cam = -R_world_to_cam @ t
    
    xyz_cam = xyz_3d @ R_world_to_cam.T + t_world_to_cam

    uvw = xyz_cam @ K_scaled.T

    z = uvw[..., 2:]
    
    xy = uvw[..., :2] / (z + 1e-6) 
    
    total_stride = vision_strides["patch_stride"] * vision_strides["conv_stride"] 

    row = (xy[..., 1] / total_stride).floor().long() 
    col = (xy[..., 0] / total_stride).floor().long() 
    
    patch_h = image_size_resize[0] // total_stride  
    patch_w = image_size_resize[1] // total_stride  
    
    valid = (z.squeeze(-1) > 0) & \
            (xy[..., 0] >= 0) & (xy[..., 0] < image_size_resize[1]) & \
            (xy[..., 1] >= 0) & (xy[..., 1] < image_size_resize[0])

    row = torch.clamp(row, 0, patch_h - 1)
    col = torch.clamp(col, 0, patch_w - 1)
    patch_idx = torch.stack([row, col], dim=-1)
    
    return patch_idx, valid





class SceneLevelContrastiveLoss(nn.Module):
    def __init__(self, token_dim, contrastive_embedding_dim=256, temperature=0.07):
        super().__init__()
        self.token_dim = token_dim
        self.contrastive_embedding_dim = contrastive_embedding_dim
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim // 2, contrastive_embedding_dim)
        )

    def forward(self, image_tokens, pointcloud_tokens):
        # (B, N, D) -> (B, D)
        img_vector_aggregated = torch.mean(image_tokens, dim=1)
        pc_vector_aggregated = torch.mean(pointcloud_tokens, dim=1)

        img_proj = self.projection_head(img_vector_aggregated) # (B, D_contrast)
        pc_proj = self.projection_head(pc_vector_aggregated)   # (B, D_contrast)

        img_proj_norm = F.normalize(img_proj, p=2, dim=-1)
        pc_proj_norm = F.normalize(pc_proj, p=2, dim=-1)

        # (B, D_contrast) @ (D_contrast, B) -> (B, B)
        logits = torch.matmul(img_proj_norm, pc_proj_norm.t()) / self.temperature

        B = image_tokens.shape[0]
        labels = torch.arange(B, dtype=torch.long, device=logits.device)

        loss_i2p = F.cross_entropy(logits, labels) # Image-to-PointCloud Loss
        loss_p2i = F.cross_entropy(logits.t(), labels) # PointCloud-to-Image Loss
        contrastive_loss = (loss_i2p + loss_p2i) / 2.0
    
        return contrastive_loss
    
class TokenLevelContrastiveLoss(nn.Module):
    def __init__(self, feature_dim, projection_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        self.image_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        self.pointcloud_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, image_features, pointcloud_features):
        B, N, _ = image_features.shape
        
        img_proj = self.image_projection_head(image_features)
        pc_proj = self.pointcloud_projection_head(pointcloud_features)

        img_proj_norm = F.normalize(img_proj, p=2, dim=-1)
        pc_proj_norm = F.normalize(pc_proj, p=2, dim=-1)

        img_proj_flat = img_proj_norm.view(B * N, -1)
        pc_proj_flat = pc_proj_norm.view(B * N, -1)

        logits = torch.matmul(img_proj_flat, pc_proj_flat.t()) / self.temperature
        labels = torch.arange(B * N, dtype=torch.long, device=logits.device)

        loss_i2p = F.cross_entropy(logits, labels)
        loss_p2i = F.cross_entropy(logits.t(), labels)
        token_contrastive_loss = (loss_i2p + loss_p2i) / 2.0
        
        return token_contrastive_loss
    
    
# image-pointcloud contrastive module
class CoordinateAwareContrastiveLoss(nn.Module):
    def __init__(self, feature_dim, projection_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.image_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        self.pointcloud_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, image_features, pointcloud_features, patch_indices, valid_mask):
        B, N_patches, D_feat = image_features.shape
        _, N_points, _ = pointcloud_features.shape
        
        img_proj = self.image_projection_head(image_features)
        pc_proj = self.pointcloud_projection_head(pointcloud_features)
        
        img_proj = F.normalize(img_proj, p=2, dim=-1)
        pc_proj = F.normalize(pc_proj, p=2, dim=-1)
        
        patch_w = int(N_patches**0.5) 
        linear_indices = patch_indices[:, :, 0] * patch_w + patch_indices[:, :, 1] # [B, N_points]
        
        D_proj = img_proj.shape[-1]
        expanded_indices = linear_indices.unsqueeze(-1).expand(-1, -1, D_proj) # [B, N_points, D_proj]
        
        target_img_proj = torch.gather(img_proj, 1, expanded_indices) # [B, N_points, D_proj]

        valid_pc_proj = pc_proj[valid_mask]             # [M, D_proj], M是整个batch中的有效点总数
        valid_target_img_proj = target_img_proj[valid_mask] # [M, D_proj]
        
        if valid_pc_proj.shape[0] == 0:
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        logits = torch.matmul(valid_pc_proj, valid_target_img_proj.t()) / self.temperature # [M, M]
        labels = torch.arange(valid_pc_proj.shape[0], device=logits.device) # 对角线是正样本
        
        loss_pc2img = F.cross_entropy(logits, labels)
        loss_img2pc = F.cross_entropy(logits.t(), labels)
        
        return (loss_pc2img + loss_img2pc) / 2
    
# tactile contrastive loss module
class TactileContrastiveLoss(nn.Module):
    def __init__(self, feature_dim, projection_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        # Define projection heads for each modality
        self.tactile_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        self.pointcloud_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        self.image_projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, tac_features, pc_features, img_features, 
                positive_pc_indices, linear_positive_img_indices):
        if tac_features.shape[0] == 0:
            return torch.tensor(0.0, device=tac_features.device, requires_grad=True)
        tac_proj = F.normalize(self.tactile_projection_head(tac_features), p=2, dim=-1)
        pc_proj = F.normalize(self.pointcloud_projection_head(pc_features), p=2, dim=-1)
        img_proj = F.normalize(self.image_projection_head(img_features), p=2, dim=-1)
        logits_tac_pc = torch.bmm(tac_proj, pc_proj.transpose(1, 2)) / self.temperature
        
        loss_tac_pc = F.cross_entropy(logits_tac_pc.view(-1, pc_proj.shape[1]), 
                                      positive_pc_indices.view(-1))

        logits_tac_img = torch.bmm(tac_proj, img_proj.transpose(1, 2)) / self.temperature

        loss_tac_img = F.cross_entropy(logits_tac_img.view(-1, img_proj.shape[1]),
                                       linear_positive_img_indices.view(-1))
        
        return (loss_tac_pc + loss_tac_img) / 2