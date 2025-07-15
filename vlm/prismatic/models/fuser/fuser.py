import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你的AttentionBlock和MLPBlock定义如下，如果不是，请替换为你的实际实现
# 例如，可以从huggingface transformers库中借用TransformerEncoderLayer的实现
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class CrossAttentionBlock(nn.Module):
    """
    一个Transformer解码器式的层，用于实现交叉注意力。
    Query Token 会注意到 Key-Value Token。
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1_q = norm_layer(embed_dim)
        self.norm1_kv = norm_layer(embed_dim) # Normalize Key/Value tokens
        
        # 直接使用 nn.MultiheadAttention
        # batch_first=True 使输入和输出的形状为 (batch, seq, feature)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        
        # Stochastic depth (DropPath)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path) # Placeholder for actual DropPath if needed

        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim, act_layer=act_layer, drop=drop)

    def forward(self, query_tokens, key_value_tokens):
        """
        Args:
            query_tokens (torch.Tensor): Shape (B, N_query, C)
            key_value_tokens (torch.Tensor): Shape (B, N_kv, C)
        Returns:
            torch.Tensor: Shape (B, N_query, C)
        """
        q_norm = self.norm1_q(query_tokens)
        kv_norm = self.norm1_kv(key_value_tokens)
        
        # MultiHeadAttention.forward: query, key, value
        # attn_output, attn_output_weights = self.attn(q_norm, kv_norm, kv_norm)
        attn_output, _ = self.attn(q_norm, kv_norm, kv_norm) # We usually don't need attention weights for the main path

        # Residual connection for query_tokens
        query_tokens = query_tokens + self.drop_path(attn_output)
        
        # MLP part
        mlp_output = self.mlp(self.norm2(query_tokens))
        query_tokens = query_tokens + self.drop_path(mlp_output)
        
        return query_tokens
    
class PerceiverAligner(nn.Module):
    def __init__(self, intermediate_dim=1024, num_queries=64, num_heads=16, 
                 num_aligner_layers=4, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        """
        使用可学习的查询 Token 来对齐和融合来自不同模态的中间表征。

        Args:
            intermediate_dim (int): 2D和3D Token被打到的中间维度，也是查询Token的维度。
            num_queries (int): 要使用的可学习查询 Token 的数量，决定了输出序列的长度。
            num_heads (int): 交叉注意力模块中的注意力头数量。
                               intermediate_dim 必须能被 num_heads 整除。
            num_aligner_layers (int): 交叉注意力块的层数。
            mlp_ratio (float): MLP 隐藏层维度的比例。
            qkv_bias (bool): 是否在注意力模块中使用偏置。
            drop (float): MLP中的dropout率。
            attn_drop (float): 注意力模块中的dropout率。
            drop_path_rate (float): 随机深度率（如果使用）。
            norm_layer (nn.Module): 归一化层。
            act_layer (nn.Module): 激活函数层。
        """
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.num_queries = num_queries

        if intermediate_dim % num_heads != 0:
            raise ValueError(f"intermediate_dim ({intermediate_dim}) must be divisible by num_heads ({num_heads})")

        # 可学习的查询Token
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, intermediate_dim))
        nn.init.normal_(self.query_tokens, std=0.02) # Common initialization

        # 构建交叉注意力层
        # Note: drop_path_rate can be implemented as a list of rates for each layer for linear increase
        # For simplicity, using a fixed one or applying it if > 0.
        # If using actual DropPath, you'd import it and use it. Here, nn.Dropout is a placeholder if drop_path > 0 in CrossAttentionBlock
        
        self.aligner_layers = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=intermediate_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rate, # Or a more sophisticated schedule for drop_path per layer
                norm_layer=norm_layer,
                act_layer=act_layer
            ) for _ in range(num_aligner_layers)
        ])

        self.final_norm = norm_layer(intermediate_dim)

    def forward(self, image_tokens_intermediate, pointcloud_tokens_intermediate):
        """
        Args:
            image_tokens_intermediate (torch.Tensor): 2D视觉Token，已投影到中间维度。
                                                      Shape: (B, N_2dtoken, intermediate_dim)
            pointcloud_tokens_intermediate (torch.Tensor): 3D视觉Token，已投影到中间维度。
                                                         Shape: (B, N_3dtoken, intermediate_dim)
        Returns:
            torch.Tensor: 融合后的查询Token表征。Shape: (B, num_queries, intermediate_dim)
        """
        B = image_tokens_intermediate.shape[0]

        # 将2D和3D token拼接起来作为交叉注意力的上下文(Key 和 Value)
        # 确保它们的特征维度 (intermediate_dim) 是一致的
        visual_context = torch.cat([image_tokens_intermediate, pointcloud_tokens_intermediate], dim=1)
        # visual_context shape: (B, N_2dtoken + N_3dtoken, intermediate_dim)

        # 扩展查询Token以匹配批次大小
        queries = self.query_tokens.expand(B, -1, -1) # Shape: (B, num_queries, intermediate_dim)

        # 依次通过所有交叉注意力层
        for layer in self.aligner_layers:
            queries = layer(query_tokens=queries, key_value_tokens=visual_context)
        
        queries = self.final_norm(queries) # 最后进行一次归一化

        return queries
    
    
class MultiModalAligner(nn.Module):
    def __init__(self, llm_token_size=4096, num_heads=12, num_cross_attn_layers=2, mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.llm_token_size = llm_token_size
        
        self.image_to_pc_cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=llm_token_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate, # Use same drop_path for simplicity
            ) for _ in range(num_cross_attn_layers)
        ])

        self.pc_to_image_cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=llm_token_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
            ) for _ in range(num_cross_attn_layers)
        ])

        self.final_norm = nn.LayerNorm(llm_token_size)
        
        for i, layer in enumerate(self.image_to_pc_cross_attn_layers):
            layer.attn = nn.MultiheadAttention(llm_token_size, num_heads, dropout=0.1, batch_first=True)
        for i, layer in enumerate(self.pc_to_image_cross_attn_layers):
            layer.attn = nn.MultiheadAttention(llm_token_size, num_heads, dropout=0.1, batch_first=True)


    def forward(self, image_tokens, pointcloud_tokens):
        fused_image_tokens = image_tokens.clone()
        fused_pc_tokens = pointcloud_tokens.clone()

        for i2p_attn, p2i_attn in zip(self.image_to_pc_cross_attn_layers, self.pc_to_image_cross_attn_layers):
            fused_image_tokens = i2p_attn(query_tokens=fused_image_tokens, key_value_tokens=fused_pc_tokens)
            
            fused_pc_tokens = p2i_attn(query_tokens=fused_pc_tokens, key_value_tokens=fused_image_tokens)

        fused_tokens = torch.cat([fused_pc_tokens, fused_image_tokens], dim=1)

        fused_tokens = self.final_norm(fused_tokens)

        return fused_tokens