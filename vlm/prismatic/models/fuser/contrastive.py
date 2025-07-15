import torch
import torch.nn as nn
import torch.nn.functional as F

# def project_3d_to_2d(xyz_3d, K, R, t, image_size, vision_strides):
#     """
#     将世界坐标系的3D点投影到2D图像，并计算对应的最终Patch索引。
    
#     Args:
#         xyz_3d: 世界坐标系的3D点 [B, N, 3]
#         K, R, t: 相机内外参
#         image_size: (H, W)，原始图像分辨率，例如 (672, 672)
#         vision_strides: 2D分词器的总步长 (patch_stride * conv_stride)
#     Returns:
#         patch_indices: 每个3D点对应的2D Patch索引 [B, N, 2] (row, col)
#         valid_mask: 是否投影在图像内 [B, N]
#     """
#     # K.to(xyz_3d.device)
#     # R.to(xyz_3d.device)
#     # t.to(xyz_3d.device)
#     H, W = image_size
#     total_stride = vision_strides['patch_stride'] * vision_strides['conv_stride'] # 例如 14 * 3 = 42
    
#     # 1. 世界坐标系 -> 相机坐标系 (这部分不变)
#     xyz_cam = torch.matmul(xyz_3d, R.t()) + t
    
#     # 2. 相机坐标系 -> 图像像素坐标系 (这部分不变)
#     xy_img = torch.matmul(xyz_cam, K.t())
#     z = xy_img[..., 2:]
#     # 检查点是否在相机前方
#     in_front_mask = z.squeeze(-1) > 0
#     xy_img = xy_img[..., :2] / (z + 1e-6)
    
#     # --- 【关键改进】 ---
#     # 3. 根据总步长计算Patch索引
#     # 直接用像素坐标除以总步长，就能得到在最终特征图上的位置
#     row = (xy_img[..., 1] / total_stride).long()  # y对应行
#     col = (xy_img[..., 0] / total_stride).long()  # x对应列
    
#     # 最终的patch网格大小
#     patch_h = H // total_stride # 672 / 42 = 16
#     patch_w = W // total_stride # 672 / 42 = 16
    
#     # 4. 检查是否在图像范围内
#     valid_mask = (in_front_mask) & \
#                  (xy_img[..., 0] >= 0) & (xy_img[..., 0] < W) & \
#                  (xy_img[..., 1] >= 0) & (xy_img[..., 1] < H)
    
#     # 将patch索引限制在有效范围内，防止越界
#     row = torch.clamp(row, 0, patch_h - 1)
#     col = torch.clamp(col, 0, patch_w - 1)
    
#     return torch.stack([row, col], dim=-1), valid_mask

def project_3d_to_2d(
    xyz_3d,                     # (B, N, 3) world 坐标
    K, R, t,                    # **原封不动**地沿用深度→点云时用的 K, R, t
    image_size_orig=(224,224),  # 深度图分辨率
    image_size_resize=(672,672),
    vision_strides={"patch_stride":14, "conv_stride":3},
):

    scale_x = image_size_resize[1] / image_size_orig[1]   # 3.0
    scale_y = image_size_resize[0] / image_size_orig[0]   # 3.0
    K_rs = K.clone()

    xyz_cam = xyz_3d @ R.T + t  

    # --- 2.  camera → pixel (在 672×672 空间) ------------------------
    uvw = xyz_cam @ K_rs.T
    z   = uvw[..., 2: ]
    xy  = uvw[..., :2] / (z + 1e-6)

    # --- 3.  生成 patch 索引 ----------------------------------------
    total_stride = vision_strides["patch_stride"] * vision_strides["conv_stride"]  # 42
    row = (xy[...,1] / total_stride).floor().long()
    col = (xy[...,0] / total_stride).floor().long()

    patch_h = image_size_resize[0] // total_stride   # 16
    patch_w = image_size_resize[1] // total_stride   # 16

    valid = (z.squeeze(-1) > 0) & \
            (xy[...,0] >= 0) & (xy[...,0] < image_size_resize[1]) & \
            (xy[...,1] >= 0) & (xy[...,1] < image_size_resize[0])

    row = torch.clamp(row, 0, patch_h - 1)
    col = torch.clamp(col, 0, patch_w - 1)
    return torch.stack([row, col], -1), valid

def project_3d_to_2d_672_pyrep_compatible(
    xyz_3d,                     # (B, N, 3) world 坐标
    K, R, t,                    # PyRep格式的相机参数
    image_size_orig=(224, 224), # 原始深度图分辨率
    image_size_resize=(672, 672), # 目标分辨率
    vision_strides={"patch_stride": 14, "conv_stride": 2},
):
    """
    基于PyRep外参格式的投影函数，支持分辨率缩放到672x672
    
    Args:
        xyz_3d: 世界坐标系的3D点 (B, N, 3)
        K, R, t: PyRep格式的相机参数
        image_size_orig: 原始图像分辨率 (H, W)
        image_size_resize: 目标分辨率 (H, W)
        vision_strides: 步长配置字典
    
    Returns:
        patch_idx: patch索引 (B, N, 2)
        valid: 有效性掩码 (B, N)
    """
    
    # 1. 计算缩放比例
    scale_x = image_size_resize[1] / image_size_orig[1]  # 672/224 = 3.0
    scale_y = image_size_resize[0] / image_size_orig[0]  # 672/224 = 3.0
    
    # 2. 调整内参矩阵以适应新分辨率
    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_x  # fx 缩放
    K_scaled[1, 1] *= scale_y  # fy 缩放  
    K_scaled[0, 2] *= scale_x  # cx 缩放
    K_scaled[1, 2] *= scale_y  # cy 缩放
    
    # 3. PyRep外参处理：计算世界到相机的变换
    R_world_to_cam = R.T
    t_world_to_cam = -R_world_to_cam @ t
    
    # 4. 转换到相机坐标系
    xyz_cam = xyz_3d @ R_world_to_cam.T + t_world_to_cam
    
    # 5. 投影到图像平面（在672x672空间）
    uvw = xyz_cam @ K_scaled.T
    z = uvw[..., 2:]
    xy = uvw[..., :2] / (z + 1e-6)
    
    # 6. 计算总步长
    total_stride = vision_strides["patch_stride"] * vision_strides["conv_stride"]  # 14 * 3 = 42
    
    # 7. 生成patch索引
    row = (xy[..., 1] / total_stride).floor().long()
    col = (xy[..., 0] / total_stride).floor().long()
    
    # 8. 计算patch网格大小
    patch_h = image_size_resize[0] // total_stride  # 672 // 42 = 16
    patch_w = image_size_resize[1] // total_stride  # 672 // 42 = 16
    
    # 9. 有效性检查（在672x672空间）
    valid = (z.squeeze(-1) > 0) & \
            (xy[..., 0] >= 0) & (xy[..., 0] < image_size_resize[1]) & \
            (xy[..., 1] >= 0) & (xy[..., 1] < image_size_resize[0])
    
    # 10. 限制索引范围
    row = torch.clamp(row, 0, patch_h - 1)
    col = torch.clamp(col, 0, patch_w - 1)
    
    # 11. 返回patch索引
    patch_idx = torch.stack([row, col], dim=-1)
    
    return patch_idx, valid


class SceneLevelContrastiveLoss(nn.Module):
    def __init__(self, token_dim, contrastive_embedding_dim=256, temperature=0.07):
        """
        计算2D图像和3D点云表征之间的场景级别对比损失。

        Args:
            token_dim (int): 输入的2D/3D token的特征维度 (即 LLM_token_size)。
            contrastive_embedding_dim (int): 对比学习投影头的输出维度。
            temperature (float): InfoNCE损失中的温度参数，用于缩放logits。
        """
        super().__init__()
        self.token_dim = token_dim
        self.contrastive_embedding_dim = contrastive_embedding_dim
        self.temperature = temperature
        
        # 投影头将聚合后的场景表征映射到用于对比学习的低维空间
        self.projection_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim // 2, contrastive_embedding_dim)
        )

    def forward(self, image_tokens, pointcloud_tokens):
        """
        计算对比损失。

        Args:
            image_tokens (torch.Tensor): 形状为 (B, N_image_token, token_dim) 的2D图像token。
            pointcloud_tokens (torch.Tensor): 形状为 (B, N_pc_token, token_dim) 的3D点云token。
        
        Returns:
            torch.Tensor: 计算得到的对比损失值 (一个标量)。
        """
        # 1. 聚合Token序列为场景级别的单一向量 (使用平均池化)
        # (B, N, D) -> (B, D)
        img_vector_aggregated = torch.mean(image_tokens, dim=1)
        pc_vector_aggregated = torch.mean(pointcloud_tokens, dim=1)

        # 2. 通过投影头得到用于对比的嵌入
        img_proj = self.projection_head(img_vector_aggregated) # (B, D_contrast)
        pc_proj = self.projection_head(pc_vector_aggregated)   # (B, D_contrast)

        # 3. L2归一化，使其成为单位向量
        img_proj_norm = F.normalize(img_proj, p=2, dim=-1)
        pc_proj_norm = F.normalize(pc_proj, p=2, dim=-1)

        # 4. 计算所有样本对的相似度矩阵 (logits)
        # (B, D_contrast) @ (D_contrast, B) -> (B, B)
        # 矩阵中 (i, j) 的值表示第 i 个图像与第 j 个点云的相似度
        logits = torch.matmul(img_proj_norm, pc_proj_norm.t()) / self.temperature

        # 5. 创建标签。对于一个批次内的样本，(i, i) 是正样本对
        B = image_tokens.shape[0]
        labels = torch.arange(B, dtype=torch.long, device=logits.device)

        # 6. 计算对称的交叉熵损失
        loss_i2p = F.cross_entropy(logits, labels) # Image-to-PointCloud Loss
        loss_p2i = F.cross_entropy(logits.t(), labels) # PointCloud-to-Image Loss

        contrastive_loss = (loss_i2p + loss_p2i) / 2.0
        
        return contrastive_loss
    
class TokenLevelContrastiveLoss(nn.Module):
    def __init__(self, feature_dim, projection_dim=256, temperature=0.07):
        """
        计算两个token序列之间的、token级别的对比损失。

        Args:
            feature_dim (int): 输入token的特征维度 (即 LLM_token_size)。
            projection_dim (int): 对比学习投影头的输出维度。
            temperature (float): InfoNCE损失中的温度参数。
        """
        super().__init__()
        self.temperature = temperature
        
        # 为两个模态分别创建投影头，因为它们的特征经过LLM前几层后可能仍有差异
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
        """
        Args:
            image_features (torch.Tensor): (B, N_tokens, D_feat)
            pointcloud_features (torch.Tensor): (B, N_tokens, D_feat)
        """
        B, N, _ = image_features.shape
        
        # 1. 通过投影头
        img_proj = self.image_projection_head(image_features)
        pc_proj = self.pointcloud_projection_head(pointcloud_features)

        # 2. L2归一化
        img_proj_norm = F.normalize(img_proj, p=2, dim=-1)
        pc_proj_norm = F.normalize(pc_proj, p=2, dim=-1)

        # 3. 为了高效计算，将token维度和batch维度融合
        # (B, N, D) -> (B * N, D)
        img_proj_flat = img_proj_norm.view(B * N, -1)
        pc_proj_flat = pc_proj_norm.view(B * N, -1)

        # 4. 计算所有token对的相似度矩阵
        # (B*N, D) @ (D, B*N) -> (B*N, B*N)
        logits = torch.matmul(img_proj_flat, pc_proj_flat.t()) / self.temperature

        # 5. 创建标签
        # 正确的配对是第i个图像token对应第i个点云token
        # 在(B*N, B*N)的矩阵中，对角线上的元素(i,i)就是正样本对
        labels = torch.arange(B * N, dtype=torch.long, device=logits.device)

        # 6. 计算对称的交叉熵损失
        loss_i2p = F.cross_entropy(logits, labels)
        loss_p2i = F.cross_entropy(logits.t(), labels)

        token_contrastive_loss = (loss_i2p + loss_p2i) / 2.0
        
        return token_contrastive_loss
    
    
    
    
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
        """
        Args:
            image_features: 2D图像特征 [B, N_patches, D] (例如 B, 576, D)
            pointcloud_features: 3D点云特征 [B, N_points, D] (例如 B, 576, D)
            patch_indices: 3D点对应的2D Patch索引 [B, N_points, 2] (row, col)
            valid_mask: 是否投影有效 [B, N_points]
        """
        B, N_patches, D_feat = image_features.shape
        _, N_points, _ = pointcloud_features.shape
        
        # 1. 投影到对比学习空间 (不变)
        img_proj = self.image_projection_head(image_features)
        pc_proj = self.pointcloud_projection_head(pointcloud_features)
        
        # 2. L2归一化 (不变)
        img_proj = F.normalize(img_proj, p=2, dim=-1)
        pc_proj = F.normalize(pc_proj, p=2, dim=-1)
        
        # 3. 获取对应的2D图像特征
        # 3.1 计算线性索引
        patch_w = int(N_patches**0.5) # 正确计算patch网格宽度
        linear_indices = patch_indices[:, :, 0] * patch_w + patch_indices[:, :, 1] # [B, N_points]
        
        # 3.2 使用gather高效地提取对应的2D特征
        # `gather`需要索引和源张量有相同的维度数
        D_proj = img_proj.shape[-1]
        expanded_indices = linear_indices.unsqueeze(-1).expand(-1, -1, D_proj) # [B, N_points, D_proj]
        
        # 从img_proj中根据索引提取目标特征
        target_img_proj = torch.gather(img_proj, 1, expanded_indices) # [B, N_points, D_proj]
        
        # 4. 只保留有效的特征对进行损失计算
        # valid_mask的形状是 [B, N_points], 我们用它来筛选
        valid_pc_proj = pc_proj[valid_mask]             # [M, D_proj], M是整个batch中的有效点总数
        valid_target_img_proj = target_img_proj[valid_mask] # [M, D_proj]
        
        if valid_pc_proj.shape[0] == 0:
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        # 5. 计算对比损失 (InfoNCE)
        # 现在我们对所有有效的特征对进行操作，负样本是batch内所有其他不匹配的对
        logits = torch.matmul(valid_pc_proj, valid_target_img_proj.t()) / self.temperature # [M, M]
        labels = torch.arange(valid_pc_proj.shape[0], device=logits.device) # 对角线是正样本
        
        loss_pc2img = F.cross_entropy(logits, labels)
        loss_img2pc = F.cross_entropy(logits.t(), labels)
        
        return (loss_pc2img + loss_img2pc) / 2
    
    
class EnhancedContrastiveLoss(nn.Module):
    def __init__(self, feature_dim, projection_dim=256, temperature=0.07,
                 # --- 【新增】用于控制新功能的参数 ---
                 enable_feature_distance_loss: bool = True,
                 distance_loss_weight: float = 0.1,
                 enable_hard_negative_mining: bool = True,
                 num_hard_negatives: int = 64
                ):
        """
        一个增强的、坐标感知的对比损失模块。

        Args:
            enable_feature_distance_loss (bool): 是否启用L1特征距离损失。
            distance_loss_weight (float): L1损失的权重。
            enable_hard_negative_mining (bool): 是否启用难负样本挖掘。
            num_hard_negatives (int): 每个正样本对要挖掘的难负样本数量。
        """
        super().__init__()
        self.temperature = temperature
        
        # --- 新功能相关的属性 ---
        self.enable_feature_distance_loss = enable_feature_distance_loss
        self.distance_loss_weight = distance_loss_weight
        self.enable_hard_negative_mining = enable_hard_negative_mining
        # 确保k值小于可能的负样本总数
        self.num_hard_negatives = num_hard_negatives

        # 投影头保持不变
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

    def _compute_info_nce(self, anchor_features, target_features):
        """
        计算InfoNCE损失, 内部支持难负样本挖掘。
        anchor_features: [M, D_proj]
        target_features: [M, D_proj]
        """
        M, _ = anchor_features.shape
        # 计算所有样本对的相似度
        logits = torch.matmul(anchor_features, target_features.t()) / self.temperature
        
        if not self.enable_hard_negative_mining:
            # --- 标准InfoNCE损失 ---
            labels = torch.arange(M, device=logits.device)
            return F.cross_entropy(logits, labels)
        else:
            # --- 带难负样本挖掘的InfoNCE损失 ---
            # 1. 提取正样本（对角线元素）
            positive_logits = logits.diag().unsqueeze(1) # [M, 1]
            
            # 2. 提取难负样本
            # 创建一个mask来忽略对角线上的正样本
            mask = ~torch.eye(M, dtype=torch.bool, device=logits.device)
            # 将对角线填充为负无穷，以便topk不会选中它们
            negative_logits = logits.masked_fill(~mask, -float('inf'))
            
            # k值不能超过负样本的总数
            k = min(self.num_hard_negatives, M - 1)
            if k <= 0: # 如果只有一个有效点，没有负样本
                return torch.tensor(0.0, device=logits.device)

            hard_negative_logits, _ = torch.topk(negative_logits, k=k, dim=1) # [M, k]
            
            # 3. 组合新的logits矩阵
            # 第0列是正样本，其余k列是难负样本
            final_logits = torch.cat([positive_logits, hard_negative_logits], dim=1) # [M, k+1]
            
            # 4. 创建新的标签，正样本总是在第0位
            labels = torch.zeros(M, dtype=torch.long, device=logits.device)
            
            return F.cross_entropy(final_logits, labels)

    def forward(self, image_features, pointcloud_features, patch_indices, valid_mask):
        B, N_patches, D_feat = image_features.shape
        _, N_points, _ = pointcloud_features.shape
        
        # --- 【关键改进：向量化实现】 ---
        
        # 1. 投影到对比学习空间 (不变)
        img_proj = self.image_projection_head(image_features)
        pc_proj = self.pointcloud_projection_head(pointcloud_features)
        
        # 2. L2归一化 (不变)
        img_proj = F.normalize(img_proj, p=2, dim=-1)
        pc_proj = F.normalize(pc_proj, p=2, dim=-1)
        
        # 3. 获取对应的2D图像特征
        # 3.1 计算线性索引
        patch_w = int(N_patches**0.5) # 正确计算patch网格宽度
        linear_indices = patch_indices[:, :, 0] * patch_w + patch_indices[:, :, 1] # [B, N_points]
        
        # 3.2 使用gather高效地提取对应的2D特征
        # `gather`需要索引和源张量有相同的维度数
        D_proj = img_proj.shape[-1]
        expanded_indices = linear_indices.unsqueeze(-1).expand(-1, -1, D_proj) # [B, N_points, D_proj]
        
        # 从img_proj中根据索引提取目标特征
        target_img_proj = torch.gather(img_proj, 1, expanded_indices) # [B, N_points, D_proj]
        
        # 4. 只保留有效的特征对进行损失计算
        # valid_mask的形状是 [B, N_points], 我们用它来筛选
        valid_pc_proj = pc_proj[valid_mask]             # [M, D_proj], M是整个batch中的有效点总数
        valid_target_img_proj = target_img_proj[valid_mask] # [M, D_proj]
        
        if valid_pc_proj.shape[0] == 0:
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        # --- 4. 计算总损失 ---
        total_loss = torch.tensor(0.0, device=image_features.device)
        
        # 4.1 计算对称的InfoNCE损失 (内部已包含难负样本挖掘逻辑)
        loss_pc2img = self._compute_info_nce(valid_pc_proj, valid_target_img_proj)
        loss_img2pc = self._compute_info_nce(valid_target_img_proj, valid_pc_proj)
        contrastive_loss = (loss_pc2img + loss_img2pc) / 2.0
        total_loss += contrastive_loss
        
        # 4.2 【可选】计算并添加特征距离损失
        if self.enable_feature_distance_loss:
            distance_loss = F.l1_loss(valid_pc_proj, valid_target_img_proj)
            total_loss += self.distance_loss_weight * distance_loss
            
        return total_loss