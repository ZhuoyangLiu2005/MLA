""" Vision Transformer (ViT) for Point Cloud Understanding in PyTorch
Hacked together by / Copyright 2020, Ross Wightman
Modified to 3D application by / Copyright 2022@Pix4Point team
"""
import logging
from typing import List
import torch
import torch.nn as nn
import math
from ..layers import create_norm
from ..layers.attention import ResidualAttentionBlock
import numpy as np
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import torch.nn.functional as F
from ..peft_module.mv_utils import PCViews
# from pointnet2_ops import pointnet2_utils
from .Point_PN import Point_PN_scan

class TokenDownsampler(nn.Module):
    def __init__(self, original_tokens=1024, target_tokens=576):
        super().__init__()
        original_side = int(original_tokens**0.5)
        target_side = int(target_tokens**0.5)
        if original_side * original_side != original_tokens or target_side * target_side != target_tokens:
            raise ValueError("Token counts must be perfect squares.")
        self.stride = original_side // target_side
        if original_side % target_side != 0:
            raise ValueError("Side length of original must be divisible by side length of target.")
        self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, tokens: torch.Tensor):
        B, L, C = tokens.shape
        H = W = int(L**0.5)
        tokens_grid = tokens.permute(0, 2, 1).view(B, C, H, W)
        downsampled_grid = self.pool(tokens_grid)
        downsampled_tokens = downsampled_grid.flatten(2).permute(0, 2, 1)
        return downsampled_tokens

def bilinear_interpolation_3d_to_1d(x, pos_embed):
    x = x.to(pos_embed.device)  # 强制统一设备
 
    x_normalized = x * (torch.tensor([76.], requires_grad=False, device=x.device) / (x.max() - x.min()))

    x_left = torch.floor(x_normalized).long()
    x_right = torch.ceil(x_normalized).long()

    x_left = x_left.clamp(0, 76)
    x_right = x_right.clamp(0, 76)

    pos_embed_left = pos_embed[x_left]
    pos_embed_right = pos_embed[x_right]

    weight_left = (x_right.float() - x_normalized).unsqueeze(-1)
    weight_right = (x_normalized - x_left.float()).unsqueeze(-1)

    interpolated_pos_embed = weight_left * pos_embed_left + weight_right * pos_embed_right
    return interpolated_pos_embed.squeeze()

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data, fps_idx

def fps_2d(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    return fps_idx.long()

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class Attention1(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mid_dim=12):
        super().__init__()
        self.num_heads = num_heads 
        self.scale = qk_scale or mid_dim ** -0.5
        self.qkv = nn.Linear(dim, mid_dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = nn.Linear(mid_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) 
        self.mid_dim = mid_dim

    def forward(self, x, mask=None):
        B, N, C = x.shape[0]//128, 128, x.shape[-1]  # B, N, C are batch size, number of points, and channel dimension respectively
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.mid_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Reshape and apply the mask
            mask = torch.where(mask, torch.tensor(-100000.0, device=mask.device), torch.tensor(0.0, device=mask.device))
            attn = attn + mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * N, self.mid_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class PointViT_ori_0615(nn.Module):
    """ Point Vision Transformer ++: with early convolutions
    """
    def __init__(self,
                 in_channels=3,
                 base_ckpt_path = "",
                 embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 add_pos_each_block=False,
                 global_feat='cls,max',
                 distill=False, 
                 adapter_args={'adapter_dim': 16, 
                             'adapter_drop_path_rate': 0.1,
                             }, 
                 attn1d_dim=0,
                 num_view=6,
                 **kwargs
                 ):
        """
        Args:
            in_channels (int): number of input channels. Default: 6. (p + rgb)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = Point_PN_scan()

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # print("self.patch_embed.out_channels: ", self.patch_embed.out_channels)
        # print("self.embed_dim: ", self.embed_dim)
        # input()
        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.add_pos_each_block = add_pos_each_block
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=self.embed_dim, n_head=num_heads, mlp_ratio=mlp_ratio,adapter_dim=adapter_args['adapter_dim'], drop_rate_adapter=adapter_args['adapter_drop_path_rate'],num_view=num_view
            ) for i in range(depth)])

        self.norm = create_norm(norm_args, self.embed_dim)  # Norm layer is extremely important here!
        self.global_feat = global_feat.split(',')
        self.out_channels = len(self.global_feat)*embed_dim
        self.distill_channels = embed_dim

        ##projection
        pc_views = PCViews()
        self.get_pos = pc_views.get_pos
        
        # load from CLIP pretrain
        base_ckpt = torch.load(base_ckpt_path)
        # print(base_ckpt.state_dict().keys())
        # input()
        self.pos_embed_1d = base_ckpt.state_dict()['positional_embedding']
        
        # load from Any2Point ckpt
        # saved_ckpt = torch.load(base_ckpt_path, map_location="cpu")
        # if 'model' not in saved_ckpt:
        #     raise KeyError(f"Not found key 'model' in {base_ckpt_path}, only have: {saved_ckpt.keys()}")
        # model_state_dict = saved_ckpt['model']
        # # 3. 直接使用已知键名 'encoder' 加载
        # if 'encoder' not in model_state_dict:
        #     # 打印所有键名辅助调试
        #     print("可用的参数键名:", model_state_dict.keys())
        #     raise KeyError(f"检查点中未找到 'encoder'")
        # self.pos_embed_1d = model_state_dict['encoder']
        
        
        self.pos_embed_1d.requires_grad = False
        ######

        if attn1d_dim!=0:
            self.attn1 = nn.ModuleList([Attention1(
                embed_dim, qkv_bias=qkv_bias, proj_drop=drop_rate, mid_dim=attn1d_dim)for i in range(num_view)])
            self.norm3 = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(num_view)])
        else:
            self.attn1=None
            self.norm3=None
        self.attn1d_dim = attn1d_dim

        self.dist_token = None
        self.n_tokens = 1
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_pos, std=.02)
        if self.dist_token is not None:
            torch.nn.init.normal_(self.dist_token, std=.02)
            torch.nn.init.normal_(self.dist_pos, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'dist_token', 'dist_token'}

    def get_num_layers(self):
        return self.depth

    def forward(self, p, x=None, coef_1dgird=2, coef_3dgird=0.08, pos_cor=[-1.0, 0.0, 1.0], args=None, target=None):
        if hasattr(p, 'keys'): 
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
            
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).float().to(self.device)  # 转换为 float32
        elif p.dtype != torch.float32:
            p = p.float()  # 强制转换为 float32
            
        center, group_input_tokens, center_idx, neighbor_idx, post_center = self.patch_embed(x,p)
        center_p, x = center, self.proj(group_input_tokens.transpose(1, 2))

        
        #####1D projection
        pos_x = self.get_pos(center_p, pos_cor)
        pos_x = pos_x.reshape(center_p.shape[0],-1,center_p.shape[1])

        interpolated_pos_embed = bilinear_interpolation_3d_to_1d(pos_x, self.pos_embed_1d)
        interpolated_pos_embed = interpolated_pos_embed.reshape(center_p.shape[0],-1,center_p.shape[1],self.embed_dim)
        interpolated_pos_embed_raw = interpolated_pos_embed.clone()

        interpolated_pos_embed = torch.mean(interpolated_pos_embed, dim=1)
        # print(interpolated_pos_embed.shape)
        # input()
        # interpolated_pos_embed = interpolated_pos_embed.squeeze()
        # print(interpolated_pos_embed.shape)
        # input()

        pos_embed = [
            self.cls_pos.expand(x.shape[0], -1, -1).to(p.device),  # 显式统一设备
            interpolated_pos_embed.to(p.device)
        ]
        # print("pos_embed[0] ", pos_embed[0].shape)
        # print("pos_embed[1] ", pos_embed[1].shape)
        # input()
        tokens = [self.cls_token.expand(x.shape[0], -1, -1), x]
        if self.dist_token is not None:
            pos_embed.insert(1, self.dist_pos.expand(x.shape[0], -1, -1)) 
            tokens.insert(1, self.dist_token.expand(x.shape[0], -1, -1)) 
        batchs = list()
        for i in range(pos_x.shape[0]):
            batchs.append(torch.full((pos_x.shape[2],), i, dtype=torch.long, device=x.device))
        batchs = torch.cat(batchs, dim=0)
        coord = center_p.reshape(-1, 3)
        start = segment_csr(coord, torch.cat([batchs.new_zeros(1), torch.cumsum(batchs.bincount(), dim=0)]),
                    reduce="min") 
        d3_grid_size = coef_3dgird

        cluster = voxel_grid(pos=coord - start[batchs], size=d3_grid_size, batch=batchs, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        mask_list = list()
        flat_grid_index_list = list()
        if 77 % coef_1dgird==0:
            grid_size = coef_1dgird
            grid_shape = (77 // grid_size)
        else:
            grid_size = coef_1dgird
            grid_shape = (77 + grid_size - 1) // grid_size 
        for i in range(pos_x.shape[1]):
            main_center = pos_x[:,i,:]
            grid_index = (main_center / grid_size).floor().long()
 
            batch_index = torch.arange(main_center.shape[0], device=main_center.device).view(-1, 1) * grid_shape
            flat_grid_index = grid_index + batch_index.expand_as(grid_index)
            flat_grid_index = flat_grid_index.view(-1)

            mask = torch.zeros((x.shape[0], x.shape[1],x.shape[1]), dtype=bool, device=x.device)
            grid_index_reshaped = flat_grid_index.reshape(x.shape[0], x.shape[1])
            mask = grid_index_reshaped[:, None, :] == grid_index_reshaped[:, :, None]
            mask = ~mask
            mask_list.append(mask)
            flat_grid_index_list.append(flat_grid_index)
        pos_embed = torch.cat(pos_embed, dim=1)
        x = torch.cat(tokens, dim=1)

        if self.add_pos_each_block:
            for block in self.resblocks:
                x = block(x, if_maxmean=self.if_maxmean)      
        else:
            x = self.pos_drop(x + pos_embed)
            for block in self.resblocks:
                x,attn_weight = block(x + pos_embed, center1=pos_x,idx_ptr=idx_ptr, 
                                      sorted_cluster_indices=sorted_cluster_indices, 
                                      cluster=cluster,grid_shape=grid_shape, mask=mask_list, 
                                      flat_grid_index=flat_grid_index_list, attn1=self.attn1, norm3=self.norm3,
                                      scale_factor=0.3,maxmean_1d=None,coef_pro=0.3)
        x = self.norm(x)
        
        return None, None, x[:, self.n_tokens:, :]

    def forward_cls_feat(self, p, x=None, args=None,target=None):  # p: p, x: features
   
        _, image_list, x = self.forward(p, x, args=args,target=target)
        token_features = x[:, self.n_tokens:, :]
        cls_feats = []
        for token_type in self.global_feat:
            if 'cls' in token_type:
                cls_feats.append(x[:, 0, :])
            elif 'max' in token_type:
                cls_feats.append(torch.max(token_features, dim=1, keepdim=False)[0])
            elif token_type in ['avg', 'mean']:
                cls_feats.append(torch.mean(token_features, dim=1, keepdim=False))
        global_features = torch.cat(cls_feats, dim=1)
        
        if self.dist_token is not None and self.training:
            return global_features, x[:, 1, :]
        else: 
            return global_features

    def forward_seg_feat(self, p, x=None):  # p: p, x: features
        p_list, x_list, x = self.forward(p, x)
        x_list[-1] = x.transpose(1, 2)
        return p_list, x_list

class PointViT_ori_0614(nn.Module):
    """ Point Vision Transformer ++: with early convolutions
    """
    def __init__(self,
                 in_channels=3,
                 base_ckpt_path = "",
                 embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 add_pos_each_block=False,
                 global_feat='cls,max',
                 distill=False, 
                 adapter_args={'adapter_dim': 16, 
                             'adapter_drop_path_rate': 0.1,
                             }, 
                 attn1d_dim=0,
                 num_view=6,
                 **kwargs
                 ):
        """
        Args:
            in_channels (int): number of input channels. Default: 6. (p + rgb)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = Point_PN_scan()

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.proj = nn.Linear(384, 768) # 注意更改
        self.add_pos_each_block = add_pos_each_block
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=self.embed_dim, n_head=num_heads, mlp_ratio=mlp_ratio,adapter_dim=adapter_args['adapter_dim'], drop_rate_adapter=adapter_args['adapter_drop_path_rate'],num_view=num_view
            ) for i in range(depth)])

        self.norm = create_norm(norm_args, self.embed_dim)  # Norm layer is extremely important here!
        self.global_feat = global_feat.split(',')
        self.out_channels = len(self.global_feat)*embed_dim
        self.distill_channels = embed_dim

        ##projection
        pc_views = PCViews()
        self.get_pos = pc_views.get_pos
        base_ckpt = torch.load(base_ckpt_path)
        self.pos_embed_1d = base_ckpt.state_dict()['positional_embedding']
        self.pos_embed_1d.requires_grad = False
        ######

        if attn1d_dim!=0:
            self.attn1 = nn.ModuleList([Attention1(
                embed_dim, qkv_bias=qkv_bias, proj_drop=drop_rate, mid_dim=attn1d_dim)for i in range(num_view)])
            self.norm3 = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(num_view)])
        else:
            self.attn1=None
            self.norm3=None
        self.attn1d_dim = attn1d_dim

        self.dist_token = None
        self.n_tokens = 1
        
        # self.token_downsampler = TokenDownsampler(original_tokens=1024, target_tokens=144)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_pos, std=.02)
        if self.dist_token is not None:
            torch.nn.init.normal_(self.dist_token, std=.02)
            torch.nn.init.normal_(self.dist_pos, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'dist_token', 'dist_token'}

    def get_num_layers(self):
        return self.depth

    def forward(self, p, x=None, coef_1dgird=2,coef_3dgird=0.08,pos_cor=[-1.0, 0.0, 1.0], args=None, target=None):
        if hasattr(p, 'keys'): 
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
            
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).float().to(self.device)  # 转换为 float32
        elif p.dtype != torch.float32:
            p = p.float()  # 强制转换为 float32
            
        center, group_input_tokens, center_idx, neighbor_idx, post_center = self.patch_embed(x,p)
        # print(group_input_tokens.shape)
        # input()
        center_p, x = center, self.proj(group_input_tokens.transpose(1, 2))

        #####1D projection
        pos_x = self.get_pos(center_p, pos_cor)
        pos_x = pos_x.reshape(center_p.shape[0],-1,center_p.shape[1])

        interpolated_pos_embed = bilinear_interpolation_3d_to_1d(pos_x, self.pos_embed_1d)
        interpolated_pos_embed = interpolated_pos_embed.reshape(center_p.shape[0],-1,center_p.shape[1],self.embed_dim)
        interpolated_pos_embed_raw = interpolated_pos_embed.clone()

        interpolated_pos_embed = torch.mean(interpolated_pos_embed, dim=1)

        pos_embed = [
            self.cls_pos.expand(x.shape[0], -1, -1).to(p.device),  
            interpolated_pos_embed.to(p.device)
        ]

        tokens = [self.cls_token.expand(x.shape[0], -1, -1), x]
        if self.dist_token is not None:
            pos_embed.insert(1, self.dist_pos.expand(x.shape[0], -1, -1)) 
            tokens.insert(1, self.dist_token.expand(x.shape[0], -1, -1)) 
        batchs = list()
        for i in range(pos_x.shape[0]):
            batchs.append(torch.full((pos_x.shape[2],), i, dtype=torch.long, device=x.device))
        batchs = torch.cat(batchs, dim=0)
        coord = center_p.reshape(-1, 3)
        start = segment_csr(coord, torch.cat([batchs.new_zeros(1), torch.cumsum(batchs.bincount(), dim=0)]),
                    reduce="min") 
        d3_grid_size = coef_3dgird

        cluster = voxel_grid(pos=coord - start[batchs], size=d3_grid_size, batch=batchs, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        mask_list = list()
        flat_grid_index_list = list()
        if 77 % coef_1dgird==0:
            grid_size = coef_1dgird
            grid_shape = (77 // grid_size)
        else:
            grid_size = coef_1dgird
            grid_shape = (77 + grid_size - 1) // grid_size 
        for i in range(pos_x.shape[1]):
            main_center = pos_x[:,i,:]
            grid_index = (main_center / grid_size).floor().long()
 
            batch_index = torch.arange(main_center.shape[0], device=main_center.device).view(-1, 1) * grid_shape
            flat_grid_index = grid_index + batch_index.expand_as(grid_index)
            flat_grid_index = flat_grid_index.view(-1)

            mask = torch.zeros((x.shape[0], x.shape[1],x.shape[1]), dtype=bool, device=x.device)
            grid_index_reshaped = flat_grid_index.reshape(x.shape[0], x.shape[1])
            mask = grid_index_reshaped[:, None, :] == grid_index_reshaped[:, :, None]
            mask = ~mask
            mask_list.append(mask)
            flat_grid_index_list.append(flat_grid_index)
        pos_embed = torch.cat(pos_embed, dim=1)
        x = torch.cat(tokens, dim=1)

        if self.add_pos_each_block:
            for block in self.resblocks:
                x = block(x, if_maxmean=self.if_maxmean)      
        else:
            x = self.pos_drop(x + pos_embed)
            for block in self.resblocks:
                x,attn_weight = block(x + pos_embed, center1=pos_x,idx_ptr=idx_ptr, 
                                      sorted_cluster_indices=sorted_cluster_indices, 
                                      cluster=cluster,grid_shape=grid_shape, mask=mask_list, 
                                      flat_grid_index=flat_grid_index_list, attn1=self.attn1, norm3=self.norm3,
                                      scale_factor=0.3,maxmean_1d=None,coef_pro=0.3)
        x = self.norm(x)
        
        return None, None, x

    def forward_cls_feat(self, p, x=None, args=None,target=None):  # p: p, x: features
   
        _, image_list, x = self.forward(p, x, args=args,target=target)
        token_features = x[:, self.n_tokens:, :]
        cls_feats = []
        for token_type in self.global_feat:
            if 'cls' in token_type:
                cls_feats.append(x[:, 0, :])
            elif 'max' in token_type:
                cls_feats.append(torch.max(token_features, dim=1, keepdim=False)[0])
            elif token_type in ['avg', 'mean']:
                cls_feats.append(torch.mean(token_features, dim=1, keepdim=False))
        global_features = torch.cat(cls_feats, dim=1)
        
        if self.dist_token is not None and self.training:
            return global_features, x[:, 1, :]
        else: 
            return global_features

    def forward_seg_feat(self, p, x=None):  # p: p, x: features
        p_list, x_list, x = self.forward(p, x)
        x_list[-1] = x.transpose(1, 2)
        return p_list, x_list
    
class PointViT(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dim=768, depth=12,
                 num_heads=6, mlp_ratio=4.,
                 target_token_count=256,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 **kwargs
            ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_embed = Point_PN_scan()
        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.proj = nn.Linear(384, 768) 
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, target_token_count + 1, self.embed_dim))

        self.norm = create_norm(norm_args, self.embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # --- 【修改】初始化新的可学习位置编码 ---
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    # _init_weights, no_weight_decay, get_num_layers 等方法保持不变
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, p, x=None, **kwargs):
        device = next(self.parameters()).device  # 获取模型所在的设备
        # 1. 初始Patch化，得到1024个patch token
        if hasattr(p, 'keys'): 
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
        # 强制转换为 float32 并移动到模型设备
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).float().to(device)
        else:
            p = p.float().to(device)
        x = x.float().to(device)  # 确保 x 也在正确设备上
            
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).float().to(self.device)  # 转换为 float32
        elif p.dtype != torch.float32:
            p = p.float()  # 强制转换为 float32
            
        center, group_input_tokens, center_idx, neighbor_idx, post_center = self.patch_embed(x,p)
        # print(f"embed: {self.embed_dim}")  # Debugging line
        # print(f"Center shape: {center.shape}")  # Debugging line
        # print(f"Group input tokens shape: {group_input_tokens.shape}")  # Debugging line
        center_p, patch_tokens = center, self.proj(group_input_tokens.transpose(1, 2))
        
        return patch_tokens, center
