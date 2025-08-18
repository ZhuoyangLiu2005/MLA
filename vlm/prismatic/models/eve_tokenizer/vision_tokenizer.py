import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable,Tuple
from PIL import Image
from torchvision import transforms

import random

from transformers import CLIPImageProcessor
from prismatic.models.constants import IMAGE_TOKEN_INDEX

class SelfImageTransform:
    def __init__(
        self,
        image_size: int = 224,
        resize_strategy: str = "resize",  # "resize" | "letterbox" | "center_crop"
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),  # CLIP 默认均值
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),   # CLIP 默认方差
    ):
        self.image_size = image_size
        self.resize_strategy = resize_strategy
        self.mean = mean
        self.std = std

        # 定义基础转换（PIL → Tensor + 归一化）
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 255] → [0.0, 1.0], HWC → CHW
            transforms.Normalize(mean=self.mean, std=self.std),  # 归一化
        ])

        # 定义 resize 策略
        if self.resize_strategy == "resize":
            self.resize = transforms.Resize((self.image_size, self.image_size))
        elif self.resize_strategy == "center_crop":
            self.resize = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
            ])
        else:
            raise ValueError(f"Unknown resize strategy: {self.resize_strategy}")

    def __call__(self, img: Image.Image, **kwargs) -> torch.Tensor:
        """输入 PIL Image，输出预处理后的 Tensor"""
        img = self.resize(img)      # 缩放/填充
        img = self.base_transform(img)  # 归一化 + 转 Tensor
        return img


class LocalAttention(nn.Module):
    def __init__(self, input_size, conv_stride, num_heads=8):
        super().__init__()
        self.conv_stride = conv_stride
        self.num_heads = num_heads
        self.scale = input_size ** -0.5

        self.q = nn.Sequential(nn.LayerNorm(input_size),
                               nn.Linear(input_size, input_size, bias=False))
        self.kv = nn.Sequential(nn.LayerNorm(input_size),
                                nn.Linear(input_size, input_size * 2, bias=False))
        self.proj = nn.Linear(input_size, input_size)

    def forward(self, features):
        reduce_features = F.avg_pool2d(features, kernel_size=self.conv_stride, stride=self.conv_stride)
        B, C, H, W = features.shape
        _, _, h, w = reduce_features.shape
        N = self.conv_stride ** 2

        reduce_features = reduce_features.flatten(2).transpose(-2, -1)
        patch_q = self.q(reduce_features).reshape(B, h * w, self.num_heads, -1).permute(0, 2, 1, 3).unsqueeze(-2)
        
        features = features.unfold(2, self.conv_stride, self.conv_stride).unfold(3, self.conv_stride, self.conv_stride)
        features = features.contiguous().view(B, C, h * w, self.conv_stride, self.conv_stride)
        patch_kv = self.kv(features.flatten(3).permute(0, 2, 3, 1))
        patch_kv = patch_kv.reshape(B, h * w, N, 2, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)

        patch_attn = (patch_q * self.scale * patch_kv[0]).sum(-1)
        patch_attn = patch_attn.softmax(dim=-1)

        aggre_features = (patch_attn.unsqueeze(-1) * patch_kv[1]).sum(-2)
        aggre_features = aggre_features.transpose(1, 2).reshape(B, h * w, -1)

        return reduce_features + self.proj(aggre_features)


class GlobalAttention(nn.Module):
    def __init__(self, input_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = input_size ** -0.5

        self.q = nn.Sequential(nn.LayerNorm(input_size),
                               nn.Linear(input_size, input_size, bias=False))
        self.kv = nn.Sequential(nn.LayerNorm(input_size),
                                nn.Linear(input_size, input_size * 2, bias=False))
        self.proj = nn.Linear(input_size, input_size)
    
    def forward(self, class_feature, features):

        B, N, C = features.shape
        class_feature = class_feature.repeat(B, 1, 1)

        patch_q, patch_kv = self.q(class_feature), self.kv(features)
        patch_q = patch_q.reshape(B, 1, self.num_heads, -1).transpose(1, 2)
        patch_kv = patch_kv.reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        patch_attn = (patch_q * self.scale * patch_kv[0]).sum(-1)
        patch_attn = patch_attn.softmax(dim=-1)

        aggre_features = (patch_attn.unsqueeze(-1) * patch_kv[1]).sum(-2)
        aggre_features = aggre_features.reshape(B, 1, -1)
        
        return class_feature + self.proj(aggre_features)
    
class MLP_GELU(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(MLP_GELU, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(1, depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class VisionTokenizer(nn.Module):
    def __init__(self, input_size, vision_tower_name):
        super().__init__()
        self.half_precision_dtype = torch.float16  
        self.is_loaded = True
        self.hidden_size = input_size
        # self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
        self.image_processor = CLIPImageProcessor(
            do_resize=True,
            size=672,
            do_center_crop=True,
            crop_size=672,
            do_normalize=True,  
            do_rescale=True,    
        )

        # patch_stride, conv_stride = self.image_processor.patch_stride, self.image_processor.conv_stride
        patch_stride = 14
        self.patch_stride = patch_stride
        self.conv_stride = 3

        self.patch_embedding = nn.Conv2d(3, input_size, kernel_size=patch_stride, stride=patch_stride, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(input_size))
        self.split_embedding = nn.Parameter(torch.randn(input_size))

        self.local_attention = LocalAttention(input_size, self.conv_stride)
        self.global_attention = GlobalAttention(input_size)

    def forward(self, pixel_values, modules):
        # if self.half_precision_dtype:
        #     pixel_values = pixel_values.to(self.half_precision_dtype)  # 将输入数据转换为半精度
        pixel_values, pixel_masks = pixel_values[:, :-1, :, :], pixel_values[:, -1:, :, :]

        patch_embeds = self.patch_embedding(pixel_values.to(dtype=self.dtype))
        patch_masks = F.avg_pool2d(pixel_masks, kernel_size=self.patch_stride, stride=self.patch_stride)
        assert len(torch.where(patch_masks % 1)[0]) == 0
        
        patch_embeds_, patch_hw_ = [], []
        for i in range(patch_embeds.shape[0]):
            if patch_masks[i, 0].sum() == 0:
                patch_embed = patch_embeds[i, :, :16, :16]
            else:
                nonzero_indices = torch.nonzero(patch_masks[i, 0], as_tuple=False)
                h1, w1 = nonzero_indices[0]
                h2, w2 = nonzero_indices[-1]
                patch_embed = patch_embeds[i, :, h1:h2+1, w1:w2+1]

            H, W = patch_embed.shape[1:]
            h, w = H // self.conv_stride, W // self.conv_stride
            patch_embed = self.local_attention(patch_embed.unsqueeze(0))
            class_embed = self.class_embedding[None, None, :].to(dtype=self.dtype)
            class_embed = self.global_attention(class_embed, patch_embed)[0]

            patch_embed = patch_embed.transpose(-2, -1).reshape(-1, h, w)
            split_embed = self.split_embedding[:, None, None].repeat(1, h, 1)

            # patch_embed = torch.cat([patch_embed, split_embed.to(dtype=self.dtype)], dim=-1)
            patch_embed = patch_embed.flatten(1).transpose(0, 1)
            patch_embeds_.append(modules(patch_embed)) # 去掉class_embed
            patch_hw_.append(torch.LongTensor([h, w]).to(self.device))

        return patch_embeds_, patch_hw_
    
    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.patch_embedding.weight.device