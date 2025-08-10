"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from prismatic.models.eve_tokenizer.vision_tokenizer import VisionTokenizer
from prismatic.models.eve_tokenizer.vision_tokenizer import MLP_GELU
from prismatic.models.fuser.contrastive import project_3d_to_2d, project_3d_to_2d_672_pyrep_compatible
from prismatic.a2pmodels.backbone.pointvit import PointViT
from prismatic.models.vlms.rec_vis import visualize_reconstruction

from action_model import ActionEmbedder, TimestepEmbedder, LabelEmbedder, FinalLayer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D  


def save_projection_visualization(
    xyz_3d, 
    patch_indices, 
    valid_mask, 
    image_size=(672, 672), 
    total_stride=42,
    save_dir="projection_visualization",
    rgb_image=None,  # 新增参数：RGB背景图像
    background_alpha=0.4  # 新增参数：背景透明度
):
    """
    
    Args:
        xyz_3d:        世界坐标系的3D点 [B, N, 3] (N=256)
        patch_indices: 投影后的Patch索引 [B, N, 2] (row, col)
        valid_mask:    有效性掩码 [B, N]
        image_size:    图像分辨率 (H, W)
        total_stride:  总步长 (e.g., 42)
        save_dir:      保存目录（自动创建）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    xyz_3d = xyz_3d[0].cpu().numpy()             # [256, 3]
    patch_indices = patch_indices[0].cpu().numpy()  # [256, 2]
    valid_mask = valid_mask[0].cpu().numpy()        # [256]
    rgb_image = rgb_image[0].float().cpu().numpy()    
    rgb_image = np.transpose(rgb_image, (1, 2, 0))

    H, W = image_size
    patch_h, patch_w = H // total_stride, W // total_stride

    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_title("2D Projection & Valid Points")
    ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
    ax1.grid(color='gray', linestyle=':', alpha=0.5)
    
    if rgb_image is not None:
        print(rgb_image)
        if rgb_image.shape[:2] == (H, W):
            ax1.imshow(rgb_image.astype(np.float32), 
                      extent=[0, W, H, 0], 
                      alpha=background_alpha)
        else:
            print(f"Warning: RGB image size {rgb_image.shape} doesn't match {image_size}")

    for x in range(0, W, total_stride):
        ax1.axvline(x, color='red', alpha=0.2)
    for y in range(0, H, total_stride):
        ax1.axhline(y, color='red', alpha=0.2)

    for i in range(len(xyz_3d)):
        color, alpha, label = ("green", 0.7, "Valid") if valid_mask[i] else ("red", 0.3, "Invalid")
        ax1.scatter(
            patch_indices[i, 1] * total_stride + total_stride//2,
            patch_indices[i, 0] * total_stride + total_stride//2,
            color=color,
            s=50,
            alpha=alpha,
            label=label if i == 0 else None,
        )
    ax1.legend(loc="upper right")
    fig1.savefig(os.path.join(save_dir, "2d_projection.png"), dpi=120, bbox_inches="tight")
    plt.close(fig1)

    heatmap = np.zeros((patch_h, patch_w), dtype=int)
    for r, c in patch_indices[valid_mask]:
        heatmap[r, c] += 1

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_title("Patch Indices Heatmap (with RGB Background)")
    
    if rgb_image is not None:
        bg_resized = rgb_image[::total_stride, ::total_stride]
        ax2.imshow(bg_resized.astype(np.float32), 
                  alpha=background_alpha, 
                  extent=[0, patch_w, patch_h, 0])
    
    im = ax2.imshow(heatmap, cmap="viridis", origin="upper", alpha=0.7)
    ax2.set_xticks(np.arange(patch_w)); ax2.set_yticks(np.arange(patch_h))
    ax2.set_xlabel("Patch Column"); ax2.set_ylabel("Patch Row")
    plt.colorbar(im, ax=ax2, label="Point Count")
    fig2.savefig(os.path.join(save_dir, "patch_heatmap_with_bg.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)

    fig3 = plt.figure(figsize=(10, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_title("3D Patch Centers")
    ax3.scatter(
        xyz_3d[:, 0], xyz_3d[:, 1], xyz_3d[:, 2],
        c='steelblue', s=30, depthshade=True
    )
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    max_range = (xyz_3d.max(axis=0) - xyz_3d.min(axis=0)).max() / 2.0
    mid = xyz_3d.mean(axis=0)
    ax3.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax3.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax3.set_zlim(mid[2] - max_range, mid[2] + max_range)
    fig3.savefig(os.path.join(save_dir, "3d_pointcloud.png"), dpi=120, bbox_inches="tight")
    plt.close(fig3)

    print(f"Visualization saved to: {os.path.abspath(save_dir)}/")


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
        recon_pointcloud: bool = False,
        num_image_recon_queries: int = 32,
        num_pointcloud_recon_queries: int = 32,
        recon_decoder_layers: int = 3,
        recon_decoder_heads: int = 8,
        image_patch_size: int = 42, 
        # === 新增MAE相关参数 ===
        mae_mask_ratio: float = 1.0,  # ROI区域内的掩码比例，1.0代表完全掩盖
        roi_dilation_kernel_size: int = 3, # ROI扩张的核大小，3代表向外扩张1圈
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
        self.mae_mask_ratio = mae_mask_ratio
        self.roi_dilation_kernel_size = roi_dilation_kernel_size
        self.image_num_patches = 256


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
            intent_decoder_layer = nn.TransformerDecoderLayer(
                d_model=token_size, nhead=decoder_heads, dim_feedforward=token_size * 2,
                dropout=0.1, activation='gelu', batch_first=True
            )
            self.intent_decoder = nn.TransformerDecoder(intent_decoder_layer, num_layers=2) 

            self.mae_mask_token = nn.Parameter(torch.zeros(1, 1, token_size))
            self.mae_pos_embed = nn.Parameter(torch.zeros(1, 256, token_size))
            mae_decoder_layer = nn.TransformerDecoderLayer(
                        d_model=token_size,
                        nhead=decoder_heads,
                        dim_feedforward=token_size * 4,
                        dropout=0.1,
                        activation='gelu',
                        batch_first=True,
                    )
            self.mae_decoder = nn.TransformerDecoder(mae_decoder_layer, num_layers=decoder_layers)
            patch_dim = self.image_patch_size ** 2 * 3
            self.mae_patch_norm = nn.LayerNorm(token_size)
            self.mae_delta_head = nn.Linear(token_size, patch_dim)
            self.mae_alpha_head = nn.Linear(token_size, 1)
            self.mae_offset_head = nn.Linear(token_size, 2)
            self.recon_delta_clip = 5  # predicted delta will be tanh(...) * delta_clip (image value scale)
            self.max_patch_shift_pixels = 8  # maximum allowed per-patch translation in pixels
            self.use_patch_offset = True  # set False to disable warping and only use delta
            
        if self.recon_pointcloud:
            self.pointcloud_recon_queries = nn.Parameter(
                torch.zeros(1, self.num_pointcloud_recon_queries, token_size)
            )
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
                "intent_decoder", "mae_decoder", "mae_patch_predictor"
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
                nn.init.normal_(self.mae_mask_token, std=0.02)
                self._init_pos_embed(self.mae_pos_embed) # 使用你已有的_init_pos_embed
                nn.init.normal_(self.mae_delta_head.weight, std=0.02)
                if self.mae_delta_head.bias is not None:
                    nn.init.constant_(self.mae_delta_head.bias, 0.0)

                # alpha: bias negative so initially model prefers copying current patch (alpha ~ sigmoid(-3)~0.047)
                nn.init.normal_(self.mae_alpha_head.weight, std=0.02)
                if self.mae_alpha_head.bias is not None:
                    nn.init.constant_(self.mae_alpha_head.bias, -3.0)

                # offset: small init
                nn.init.normal_(self.mae_offset_head.weight, std=0.001)
                if self.mae_offset_head.bias is not None:
                    nn.init.constant_(self.mae_offset_head.bias, 0.0)
                
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
    
    def load_encoder_to_vision_tower(self, ckpt_path, vision_tower_3d):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_state = checkpoint['model']
        encoder_state = {
            k.replace('encoder.', ''): v 
            for k, v in model_state.items() 
            if k.startswith('encoder.')
        }
        
        if not encoder_state:
            raise KeyError(f"parameter 'encoder' didn't found, can use: {list(model_state.keys())[:5]}")

        if hasattr(vision_tower_3d, 'module'): 
            vision_tower = vision_tower_3d.module
        else:
            vision_tower = vision_tower_3d

        missing_keys, unexpected_keys = vision_tower.load_state_dict(encoder_state, strict=False)
        
        print(f"successfully loaded {len(encoder_state)} parameters to vision_tower_3d")
        if missing_keys:
            print(f"! 缺失参数 ({len(missing_keys)} 个):\n  {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"! 多余参数 ({len(unexpected_keys)} 个):\n  {unexpected_keys[:3]}...")

        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        freeze_weights: bool = True,
        class_dropout_prob: float = 0.0,
        use_diff: bool = False,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            class_dropout_prob=class_dropout_prob,
            use_diff=use_diff,
            **kwargs,
        )

        if not isinstance(pretrained_checkpoint, dict):
            # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        else:
            model_state_dict = pretrained_checkpoint
        
        assert (
            "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `llm_backbone`!"

        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"],strict=False) # not strict, because we added contrastive loss module
        
        vlm.load_encoder_to_vision_tower(ckpt_path="/media/liuzhuoyang/new_vla/Any2Point/Any2Point_CLIP_Lang/ckpts/Language_CLIP_Scan.pth",
            vision_tower_3d=vlm.vision_tower_3d)
        
        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """
        if stage == "align":
            self.vision_tower_2d.requires_grad_(False)
            self.vision_tower_3d.requires_grad_(False)
            self.projector_2d.requires_grad_(True)
            self.projector_3d.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["proprio_embedder","projector_3d","projector_2d"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 2D ", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 3D ", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 2D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 3D ", ctx_level=1)

        elif stage in {"finetune", "vla-train"}:
            self.vision_tower_2d.requires_grad_(False)
            self.vision_tower_3d.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector_2d.requires_grad_(True)
            self.projector_3d.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone","projector_2d","projector_3d",
                                          "proprio_embedder"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder","final_layer"])

            if self.use_reconstruction:
                self.trainable_module_keys.extend(self._get_reconstruction_module_keys())

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 2D ", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 3D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 2D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 3D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Reconstruct modules ", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            self.vision_tower_2d.requires_grad_(True)
            self.vision_tower_3d.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector_2d.requires_grad_(True)
            self.projector_3d.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_tower_2d", "vision_tower_3d", 
                                          "projector_2d", "projector_3d",
                                          "llm_backbone", "proprio_embedder"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.use_reconstruction:
                self.trainable_module_keys.extend(self._get_reconstruction_module_keys())

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Tower 2D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Tower 3D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 2D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 3D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Reconstruct modules ", ctx_level=1)

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vision_tower_2d.requires_grad_(False)
            self.vision_tower_3d.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector_2d.requires_grad_(False)
            self.projector_3d.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone", "proprio_embedder"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Tower 2D ", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Tower 3D ", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector 2D ", ctx_level=1)
            overwatch.info(f"[Frozen]                    🥶   =>> Projector 3D ", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            self.vision_tower_2d.requires_grad_(True)
            self.vision_tower_3d.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_tower_2d", "vision_tower_3d", "llm_backbone", "proprio_embedder"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Tower 2D ", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Tower 3D ", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            # fmt: on

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={PointViT, VisionTokenizer},
        )
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={MLP_GELU},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )
    
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
        # print("projected_image_tokens.shape: ", projected_image_tokens.shape)
        # print("projected_pointcloud_tokens.shape: ", projected_pointcloud_tokens.shape)
        # print("pointcloud_centers.shape: ", pointcloud_centers.shape)
        # input("Press Enter to continue...")
        
        patch_indices, valid_mask = project_3d_to_2d_672_pyrep_compatible(
            pointcloud_centers, 
            camera_params['K'].to(pointcloud_centers.device),
            camera_params['R'].to(pointcloud_centers.device),
            camera_params['t'].to(pointcloud_centers.device),
            image_size_resize=(672, 672),
            vision_strides={'patch_stride': 14, 'conv_stride': 3}
        )
        
        ## project visualization
        # save_projection_visualization(
        #     pointcloud_centers, 
        #     patch_indices, 
        #     valid_mask, 
        #     save_dir="/media/liuzhuoyang/new_vla/5D_VLA_beta/vis", 
        #     rgb_image=images,
        # )
        # input("Press Enter to continue...")
        
        N_pc = projected_pointcloud_tokens.shape[1]
        N_img = projected_image_tokens.shape[1]
        assert N_pc == N_img # assert that the number of tokens are equal
        
        projected_fused_tokens = torch.cat(
            [projected_pointcloud_tokens, projected_image_tokens], 
            dim=1
        )

        return projected_fused_tokens, patch_indices, valid_mask

    def _dilate_mask(self, mask, kernel_size):
        """Dilate a boolean mask of shape [B, H, W] using max_pool2d safely."""
        # mask: [B, H, W] (bool)
        assert mask.dim() == 3, "Expected mask shape [B,H,W]"
        padding = (kernel_size - 1) // 2
        # Make it 4D: [B, C=1, H, W]
        m = mask.float().unsqueeze(1)  # [B,1,H,W]
        dil = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=padding)
        dil = (dil > 0.0).squeeze(1)   # [B,H,W] bool
        return dil

    def _create_roi_mask_from_indices(self, patch_indices: torch.Tensor) -> torch.Tensor:
        """
        根据点云投影到的2D patch坐标，创建一个[B, H, W]的ROI掩码。

        Args:
            patch_indices (torch.Tensor): 形状为 [B, N_points, 2]，
                                        其中 N_points 是点云patch数量(256),
                                        维度2是(y, x)坐标。

        Returns:
            torch.Tensor: 形状为 [B, H_patch, W_patch] 的布尔掩码, e.g., [B, 16, 16]。
        """
        batch_size = patch_indices.shape[0]
        patch_grid_size = int(self.image_num_patches**0.5) # e.g., 16
        
        # 创建一个全为False的基础掩码
        roi_mask = torch.zeros(
            batch_size, 
            patch_grid_size, 
            patch_grid_size, 
            dtype=torch.bool, 
            device=patch_indices.device
        )

        # 使用高效的向量化索引来标记所有被点云投影覆盖的patch
        # 1. 创建批次索引
        batch_idx = torch.arange(batch_size, device=patch_indices.device).view(batch_size, 1)
        
        # 2. 提取y和x坐标
        y_coords = patch_indices[..., 0]  # 所有点的y坐标
        x_coords = patch_indices[..., 1]  # 所有点的x坐标
        
        # 3. 在对应的 [batch, y, x] 位置上设置为True
        # 这个操作会处理掉重复的坐标，并将所有出现过的坐标位置都标记出来
        roi_mask[batch_idx, y_coords, x_coords] = True
        
        return roi_mask
    
    def reconstruct_modalities(self, 
                               llm_hidden_states, 
                               current_image_features,
                               current_images_patches,
                               roi_mask_2d,  
                               batch_size,
                               use_roi: bool = True
                            ):
        reconstruction_outputs = {}
        memory = llm_hidden_states  # [B, seq_len, token_size]
        
        if self.recon_image:
            # intent_queries = self.image_recon_queries.expand(batch_size, -1, -1)
            # intent_features = self.intent_decoder(tgt=intent_queries, memory=memory)

            # # robust dilation
            # reconstruction_roi_mask_2d = self._dilate_mask(roi_mask_2d, self.roi_dilation_kernel_size)
            # reconstruction_roi_mask = reconstruction_roi_mask_2d.view(batch_size, -1)  # [B, num_patches] bool

            # # Prepare decoder input: replace content at mask positions with mask token, then add pos
            # decoder_input_tokens = current_image_features.clone()  # [B, num_patches, token_size]
            # mask_token_vec = self.mae_mask_token.view(-1)
            # decoder_input_tokens[reconstruction_roi_mask] = mask_token_vec
            # decoder_input_tokens = decoder_input_tokens + self.mae_pos_embed

            # # run decoder
            # reconstructed_features = self.mae_decoder(tgt=decoder_input_tokens, memory=intent_features)  # [B, num_patches, token_size]

            # # Predict delta/alpha/offset FOR ALL PATCHES (so gradients flow globally)
            # features_flat = reconstructed_features.view(-1, reconstructed_features.shape[-1])  # [B*num_patches, token_size]
            # features_norm = self.mae_patch_norm(features_flat)
            # delta_all = self.mae_delta_head(features_norm)  # [B*num_patches, patch_dim]
            # alpha_all = torch.sigmoid(self.mae_alpha_head(features_norm).squeeze(-1))  # [B*num_patches]
            # offset_all = self.mae_offset_head(features_norm)  # [B*num_patches, 2]

            # # Reshape back
            # B, num_patches, _ = reconstructed_features.shape
            # patch_dim = self.image_patch_size ** 2 * 3
            # delta_all = delta_all.view(B, num_patches, patch_dim)
            # alpha_all = alpha_all.view(B, num_patches)  # in [0,1]
            # offset_all = offset_all.view(B, num_patches, 2)

            # # Apply scaling/clipping (assume images are standardized; delta_clip is relative scale)
            # delta_all = torch.tanh(delta_all) * self.recon_delta_clip
            # offset_all = torch.tanh(offset_all) * float(self.max_patch_shift_pixels)

            # # current images patches: [B, num_patches, patch_dim]
            # curr_patches = current_images_patches  # already provided
            # # reshape to image patches for optional warping
            # C = 3
            # ps = self.image_patch_size
            # curr_patches_img = curr_patches.view(B * num_patches, C, ps, ps)
            # offset_all_img = offset_all.view(B * num_patches, 2)

            # # If using offsets, do warp per-patch (but grid_sample needs float32)
            # if self.use_patch_offset:
            #     # build affine matrices for each patch
            #     tx = offset_all_img[:, 0]  # pixels
            #     ty = offset_all_img[:, 1]
            #     tx_norm = 2.0 * tx / float(ps - 1)
            #     ty_norm = 2.0 * ty / float(ps - 1)

            #     affines = torch.zeros(B * num_patches, 2, 3, device=offset_all_img.device, dtype=offset_all_img.dtype)
            #     affines[:, 0, 0] = 1.0
            #     affines[:, 1, 1] = 1.0
            #     affines[:, 0, 2] = tx_norm
            #     affines[:, 1, 2] = ty_norm

            #     # grid_sample doesn't support bfloat16 -> cast to float for grid_sample then cast back
            #     curr_patches_img_f = curr_patches_img.float()
            #     grid = F.affine_grid(affines.float(), size=(B * num_patches, C, ps, ps), align_corners=True)
            #     warped = F.grid_sample(curr_patches_img_f, grid, mode='bilinear', padding_mode='border', align_corners=True)
            #     warped = warped.to(curr_patches_img.dtype)  # keep original dtype (maybe bfloat16)
            # else:
            #     warped = curr_patches_img

            # # apply delta (delta_all reshape)
            # delta_img = delta_all.view(B * num_patches, C, ps, ps)

            # # gen_weight = 0.95
            # # pure_pred = delta_img
            # # residual_pred = curr_patches_img + delta_img
            # # roi_pred = (1 - gen_weight) * residual_pred + gen_weight * pure_pred
            # # non_roi_pred = warped + delta_img
            # # roi_mask_flat = reconstruction_roi_mask.view(B * num_patches, 1, 1, 1)  # 方便广播
            # # predicted_img_patches = torch.where(
            # #     roi_mask_flat,
            # #     roi_pred,       # ROI 内
            # #     non_roi_pred    # ROI 外
            # # )

            # # # roi_mask_flat = reconstruction_roi_mask.view(B * num_patches, 1, 1, 1)
            # # # predicted_img_patches = torch.where(
            # # #     roi_mask_flat.bool(),
            # # #     delta_img,               # ROI 内
            # # #     warped + delta_img       # ROI 外
            # # # )

            # # # predicted_img_patches = warped + delta_img  # still same dtype as warped/delta (watch dtype)

            # # # blend with curr by alpha (alpha_all broadcast)
            # # alpha_all = torch.where(
            # #     reconstruction_roi_mask,              # 条件: ROI 位置
            # #     torch.ones_like(alpha_all),            # ROI 内 -> 1.0
            # #     alpha_all                              # ROI 外 -> 保留预测值
            # # )
            # # alpha_img = alpha_all.view(B * num_patches, 1, 1, 1)
            # # blended = alpha_img * predicted_img_patches + (1.0 - alpha_img) * curr_patches_img


            # # without roi
            # gen_weight = 0.95
            # pure_pred = delta_img
            # residual_pred = curr_patches_img + delta_img
            # predicted_img_patches = (1 - gen_weight) * residual_pred + gen_weight * pure_pred  # 全图用ROI内部逻辑

            # # alpha 全部设为 1
            # alpha_all = torch.ones_like(alpha_all)

            # # blend
            # alpha_img = alpha_all.view(B * num_patches, 1, 1, 1)
            # blended = alpha_img * predicted_img_patches + (1.0 - alpha_img) * curr_patches_img



            # # reshape back to [B, num_patches, patch_dim]
            # blended_flat = blended.view(B, num_patches, -1)

            # # Compose reconstructed_patches: for non-ROI we can still prefer copying current (but keep grads through delta)
            # # We will keep final prediction as blended everywhere, but loss can focus on ROI
            # reconstructed_patches = blended_flat

            # reconstruction_outputs['image_reconstruction'] = reconstructed_patches
            # reconstruction_outputs['reconstruction_roi_mask'] = reconstruction_roi_mask
            # reconstruction_outputs['delta_all'] = delta_all

            intent_queries = self.image_recon_queries.expand(batch_size, -1, -1)
            intent_features = self.intent_decoder(tgt=intent_queries, memory=memory)

            # decoder input (全图)
            decoder_input_tokens = current_image_features.clone()
            mask_token_vec = self.mae_mask_token.view(-1)
            decoder_input_tokens = decoder_input_tokens + self.mae_pos_embed

            # run decoder
            reconstructed_features = self.mae_decoder(
                tgt=decoder_input_tokens, 
                memory=intent_features
            )

            # predict delta/alpha/offset (全图)
            features_flat = reconstructed_features.view(-1, reconstructed_features.shape[-1])
            features_norm = self.mae_patch_norm(features_flat)
            delta_all = self.mae_delta_head(features_norm)
            alpha_all = torch.sigmoid(self.mae_alpha_head(features_norm).squeeze(-1))
            offset_all = self.mae_offset_head(features_norm)

            # reshape back
            B, num_patches, _ = reconstructed_features.shape
            patch_dim = self.image_patch_size ** 2 * 3
            delta_all = delta_all.view(B, num_patches, patch_dim)
            alpha_all = alpha_all.view(B, num_patches)
            offset_all = offset_all.view(B, num_patches, 2)

            # clip
            delta_all = torch.tanh(delta_all) * self.recon_delta_clip
            offset_all = torch.tanh(offset_all) * float(self.max_patch_shift_pixels)

            # prepare patch images
            C = 3
            ps = self.image_patch_size
            curr_patches_img = current_images_patches.view(B * num_patches, C, ps, ps)

            # offset warp if needed
            if self.use_patch_offset:
                tx = offset_all.view(-1, 2)[:, 0]
                ty = offset_all.view(-1, 2)[:, 1]
                tx_norm = 2.0 * tx / float(ps - 1)
                ty_norm = 2.0 * ty / float(ps - 1)
                affines = torch.zeros(B * num_patches, 2, 3, device=offset_all.device, dtype=offset_all.dtype)
                affines[:, 0, 0] = 1.0
                affines[:, 1, 1] = 1.0
                affines[:, 0, 2] = tx_norm
                affines[:, 1, 2] = ty_norm
                curr_patches_img_f = curr_patches_img.float()
                grid = F.affine_grid(affines.float(), size=(B * num_patches, C, ps, ps), align_corners=True)
                warped = F.grid_sample(curr_patches_img_f, grid, mode='bilinear', padding_mode='border', align_corners=True)
                warped = warped.to(curr_patches_img.dtype)
            else:
                warped = curr_patches_img

            # delta image
            delta_img = delta_all.view(B * num_patches, C, ps, ps)

            # 全图统一重建逻辑
            gen_weight = 0.95
            pure_pred = delta_img
            residual_pred = curr_patches_img + delta_img
            predicted_img_patches = (1 - gen_weight) * residual_pred + gen_weight * pure_pred

            # alpha 全 1
            alpha_all = torch.ones_like(alpha_all)
            alpha_img = alpha_all.view(B * num_patches, 1, 1, 1)
            blended = alpha_img * predicted_img_patches + (1.0 - alpha_img) * curr_patches_img

            reconstructed_patches = blended.view(B, num_patches, -1)

            # outputs（不再返回 ROI mask）
            reconstruction_outputs['image_reconstruction'] = reconstructed_patches
            reconstruction_outputs['delta_all'] = delta_all
            reconstruction_outputs['alpha_all'] = alpha_all

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
    
    def compute_reconstruction_losses(self, 
        reconstruction_outputs, 
        next_images=None, 
        next_point_cloud=None,
        current_point_cloud=None,
    ):
        losses = {}
        total_loss = 0.0
        
        if self.recon_image and 'image_reconstruction' in reconstruction_outputs and next_images is not None:
            # reconstructed_patches = reconstruction_outputs['image_reconstruction']  # [B,256, patch_dim]
            # reconstruction_roi_mask = reconstruction_outputs['reconstruction_roi_mask']  # [B,256] bool

            # next_images_patches = self._images_to_patches(next_images)  # [B,256,patch_dim]
            # curr_patches = None  # you have current_images_patches in calling scope; but recompute if needed

            # pred_roi = reconstructed_patches[reconstruction_roi_mask]
            # gt_roi = next_images_patches[reconstruction_roi_mask]

            # if pred_roi.numel() > 0:
            #     # main reconstruction objective (use both L2 and L1)
            #     recon_mse = F.mse_loss(pred_roi, gt_roi)
            #     recon_l1 = F.l1_loss(pred_roi, gt_roi)
            #     image_recon_loss = recon_mse + 0.5 * recon_l1
            #     losses['image_roi_reconstruction_loss'] = image_recon_loss
            #     total_loss = total_loss + image_recon_loss

            # # small background consistency loss to discourage absurd deltas outside ROI
            # # choose small weight like 0.01
            # bg_mask = ~reconstruction_roi_mask
            # pred_bg = reconstructed_patches[bg_mask]
            # gt_bg = next_images_patches[bg_mask]
            # if pred_bg.numel() > 0:
            #     bg_l1 = F.l1_loss(pred_bg, gt_bg)
            #     losses['bg_consistency_loss'] = 0.01 * bg_l1
            #     total_loss = total_loss + losses['bg_consistency_loss']
            # if 'delta_all' in reconstruction_outputs:
            #     # reconstruction_outputs['delta_all']: [B, num_patches, patch_dim]
            #     delta_norm = reconstruction_outputs['delta_all'].abs().mean()
            #     delta_loss = -0.1 * delta_norm  
            #     losses['delta_magnitude_reward'] = delta_loss
            #     total_loss = total_loss + delta_loss
            reconstructed_patches = reconstruction_outputs['image_reconstruction']  # [B, num_patches, patch_dim]
            next_images_patches = self._images_to_patches(next_images)  # [B, num_patches, patch_dim]

            # === 全图重建 loss（原来 ROI 内的逻辑） ===
            recon_mse = F.mse_loss(reconstructed_patches, next_images_patches)
            recon_l1 = F.l1_loss(reconstructed_patches, next_images_patches)
            image_recon_loss = recon_mse + 0.5 * recon_l1
            losses['image_reconstruction_loss'] = image_recon_loss
            total_loss = total_loss + image_recon_loss

            # === delta magnitude reward 保留 ===
            if 'delta_all' in reconstruction_outputs:
                # reconstruction_outputs['delta_all']: [B, num_patches, patch_dim]
                delta_norm = reconstruction_outputs['delta_all'].abs().mean()
                delta_loss = -0.1 * delta_norm  
                losses['delta_magnitude_reward'] = delta_loss
                total_loss = total_loss + delta_loss

        def chamfer_distance(pred, gt):
            # pred/gt: [B, N, 3]
            dist = torch.cdist(pred, gt)  # [B, N, M]
            loss = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
            return loss

        def earth_movers_distance(pred, gt):
            # Flatten batch dimension for pairwise distance computation
            B, N, _ = pred.shape
            pred_flat = pred.view(-1, 3)  # [B*N, 3]
            gt_flat = gt.view(-1, 3)      # [B*N, 3]
            
            dist_matrix = torch.cdist(pred_flat, gt_flat)  # [B*N, B*N]
            
            dist_matrix = dist_matrix.view(B, N, B, N)
            
            emd_loss = 0.0
            for i in range(B):
                # Hungarian algorithm would be better but computationally expensive
                # Here we use a simplified approximation
                row_min = dist_matrix[i, :, i, :].min(dim=1)[0].mean()
                col_min = dist_matrix[i, :, i, :].min(dim=0)[0].mean()
                emd_loss += (row_min + col_min) / 2
                
            return emd_loss / B

        if self.recon_pointcloud and next_point_cloud is not None:
            if 'pointcloud_coord_reconstruction' in reconstruction_outputs:
                # Extract coordinates from ground truth
                assert next_point_cloud.shape[2] == 3, "Point cloud must have 3 dimensions (XYZ)"

                pc_coord_pred = reconstruction_outputs['pointcloud_coord_reconstruction']
                # pc_coord_loss = 0.1 * chamfer_distance(pc_coord_pred, next_point_cloud) 
                pc_coord_loss = earth_movers_distance(pc_coord_pred, next_point_cloud) 
                losses['pointcloud_coord_loss'] = pc_coord_loss
                total_loss += pc_coord_loss

        losses['total_reconstruction_loss'] = total_loss
        return losses
    
    
    def _images_to_patches(self, images):
        B, C, H, W = images.shape
        patch_size = self.image_patch_size
        
        assert C == 3, f"Expected 3 channels (RGB), got {C}"
        assert H == W == 672, f"Expected 672x672 image, got {H}x{W}"
        
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # [B, C=3, num_patches_h=16, num_patches_w=16, patch_h=42, patch_w=42]
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size) # [B, C=3, num_patches=256, patch_h=42, patch_w=42]
        patches = patches.permute(0, 2, 1, 3, 4).contiguous() # [B, num_patches=256, C=3, patch_h=42, patch_w=42]
        patches = patches.view(B, -1, C * patch_size * patch_size) # [B, num_patches=256, C*patch_h*patch_w=5292]
        
        return patches
    
    def _patches_to_images(self, patches):
        B, num_patches, patch_dim = patches.shape
        patch_size = self.image_patch_size
        
        expected_patch_dim = 3 * patch_size * patch_size
        assert patch_dim == expected_patch_dim, f"Expected patch_dim={expected_patch_dim}, got {patch_dim}"
        assert num_patches == 256, f"Expected 256 patches, got {num_patches}"
        
        patches = patches.view(B, num_patches, 3, patch_size, patch_size) # [B, 256, 3, 42, 42]
        num_patches_h = num_patches_w = int(num_patches ** 0.5)  # 16
        patches = patches.view(B, num_patches_h, num_patches_w, 3, patch_size, patch_size)
        images = patches.permute(0, 3, 1, 4, 2, 5).contiguous() # [B, 3, 16, 42, 16, 42]
        images = images.view(B, 3, num_patches_h * patch_size, num_patches_w * patch_size) # [B, 3, 672, 672]
        
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
            
        # Convert to bfloat16 if needed
        if proprio is not None:
            proprio = proprio.to(torch.bfloat16)
        if x is not None:
            x = x.to(torch.bfloat16)
        if t is not None:
            t = t.to(torch.bfloat16)
        if z is not None:
            z = z.to(torch.bfloat16)
        
        if self.training:
            tag_0, tag_1 = 2, 0
            tag_2 = 3 ## EOD(32002) + _(29871) + 7dof + EOS(2)
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
        
        
        # get image and point_cloud embeddings
        point_cloud = point_cloud.to(self.device) 
        projected_fused_embeddings, patch_indices, valid_mask= self.get_fused_tokens(images, point_cloud) 
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        
        # Create initial embeddings with fused tokens
        z = torch.cat([input_embeddings[:, :1, :], 
                       projected_fused_embeddings, 
                       input_embeddings[:, 1:, :]], dim=1
                    )
        
        # Token layout tracking
        N_pc = projected_fused_embeddings.shape[1] // 2
        N_img = projected_fused_embeddings.shape[1] // 2
        bos_token_len = 1
        pc_tokens_start_idx = bos_token_len
        pc_tokens_end_idx = pc_tokens_start_idx + N_pc
        img_tokens_start_idx = pc_tokens_end_idx
        img_tokens_end_idx = img_tokens_start_idx + N_img
        current_image_features = projected_fused_embeddings[:, N_pc:, :]

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
        multimodal_attention_mask = torch.cat(multimodal_attention_mask, dim=0) if len(multimodal_attention_mask) !=0 else None
        multimodal_labels = torch.cat(multimodal_labels, dim=0) if len(multimodal_labels) !=0 else None

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
            current_images_patches = self._images_to_patches(images[:, :3, :, :])
            roi_mask_2d = self._create_roi_mask_from_indices(patch_indices)
            # Perform multimodal reconstruction
            reconstruction_outputs = self.reconstruct_modalities(
                llm_hidden_states, 
                current_image_features=current_image_features,
                current_images_patches=current_images_patches,
                roi_mask_2d=roi_mask_2d,
                batch_size=fused_embeddings.shape[0],
            )
            
            assert next_images is not None and next_point_cloud is not None, "Reconstruction requires ground truth images and point clouds!"
            reconstruction_losses = self.compute_reconstruction_losses(
                reconstruction_outputs, 
                next_images=next_images, 
                next_point_cloud=next_point_cloud,
                current_point_cloud=point_cloud,  # Pass current frame for movement analysis
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
            
            # # Return with reconstruction information
            if self.training:
                visualize_reconstruction(reconstruction_outputs, 
                    next_images, next_point_cloud, 
                    "/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vis/recon_vis_2dmaeall"
                )
                # self.visualize_reconstruction_img(
                #     reconstruction_outputs, 
                #     next_images,
                #     "/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vis/recon_vis_2donlyMAE"
                # )
                # self.visualize_incremental_reconstruction(
                #     reconstruction_outputs, 
                #     images,
                #     next_images,
                #     "/media/liuzhuoyang/new_vla/Rec_Diff_beta/LLM_policy/vis/recon_vis_2donlyMAE"
                # )
                return output, noise_pred, reconstruction_outputs, reconstruction_losses
            else:
                return output, noise_pred
        
        # # Return with reconstruction information
        if self.use_reconstruction and self.training:
            return output, reconstruction_outputs, reconstruction_losses
        else:
            return output # autoregressive
        
    def visualize_incremental_reconstruction(self, reconstruction_outputs, current_images, next_images, save_dir):
        """可视化增量重建结果，帮助调试"""
        import os
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)
        
        if current_images is None or 'image_reconstruction' not in reconstruction_outputs:
            return
        
        batch_size = current_images.shape[0]
        for b in range(min(batch_size, 1)):  # 只可视化前4个样本
            
            # 转换图像格式用于显示
            current_img = current_images[:, :3, :, :][b].cpu().float().numpy().transpose(1, 2, 0)  # CHW -> HWC
            next_img = next_images[b].cpu().float().numpy().transpose(1, 2, 0) if next_images is not None else None
            
            # 重建图像
            reconstructed_patches = reconstruction_outputs['image_reconstruction'][b:b+1]  # [1, N_patches, patch_dim]
            reconstructed_img = self._patches_to_images(reconstructed_patches)[0]  # [C, H, W]
            reconstructed_img = reconstructed_img.cpu().float().detach().numpy().transpose(1, 2, 0)  # CHW -> HWC
            
            if 'change_masks' in reconstruction_outputs:
                change_mask = reconstruction_outputs['change_masks'][b].cpu().float().detach()  # [N_patches, 1]
                change_mask_patches = change_mask.reshape(1, -1, 1).repeat_interleave(self.image_patch_size**2 * 3, dim=2)
                change_mask_img = self._patches_to_images(change_mask_patches)[0, 0].cpu().float().numpy()
            else:
                change_mask_img = None
            
            # 创建对比图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            axes[0].imshow(current_img)
            axes[0].set_title('Current Frame')
            axes[0].axis('off')
            
            if next_img is not None:
                axes[1].imshow(next_img)
                axes[1].set_title('Next Frame (GT)')
                axes[1].axis('off')
            
            axes[2].imshow(reconstructed_img)
            axes[2].set_title('Reconstructed Next Frame')
            axes[2].axis('off')
            
            if change_mask_img is not None:
                axes[3].imshow(change_mask_img, cmap='hot')
                axes[3].set_title('Change Mask')
                axes[3].axis('off')
            
            # 差异图
            if next_img is not None:
                diff_img = np.abs(reconstructed_img - next_img)
                axes[4].imshow(diff_img)
                axes[4].set_title('Reconstruction Error')
                axes[4].axis('off')
            
            # 当前帧和下一帧的差异
            if next_img is not None:
                real_diff = np.abs(next_img - current_img)
                axes[5].imshow(real_diff)
                axes[5].set_title('Real Change')
                axes[5].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/reconstruction_sample_{b}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
    def visualize_reconstruction_img(self, reconstruction_outputs, next_images, save_dir):
        """8-subplot layout with Target/Recon RGB + their separate channels"""
        import os
        import matplotlib.pyplot as plt
        from torchvision.utils import save_image
        
        os.makedirs(save_dir, exist_ok=True)
        
        if 'image_reconstruction' in reconstruction_outputs and next_images is not None:
            # Convert patches to full images
            patch_pred = reconstruction_outputs['image_reconstruction']  # [B, num_patches, patch_dim]
            reconstructed_images = self._patches_to_images(patch_pred)  # [B, 3, 672, 672]

            # Clamp values to [0,1] range
            # reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
            # next_images = torch.clamp(next_images, 0, 1)
            
            batch_size = min(1, reconstructed_images.shape[0])  # Only visualize first sample
            
            for i in range(batch_size):
                # Create 2x4 subplots
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                fig.suptitle('Image Reconstruction Comparison', fontsize=16)

                # --- Row 1: Target ---
                # Target RGB
                target_img = next_images[i].cpu().float().detach().permute(1, 2, 0).numpy()
                axes[0, 0].imshow(target_img)
                axes[0, 0].set_title('Target (RGB)')
                axes[0, 0].axis('off')

                # Target R/G/B Channels
                target_tensor = next_images[i].cpu().float().detach()
                axes[0, 1].imshow(target_tensor[0], cmap='Reds', vmin=0, vmax=1)
                axes[0, 1].set_title('Target (Red)')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(target_tensor[1], cmap='Greens', vmin=0, vmax=1)
                axes[0, 2].set_title('Target (Green)')
                axes[0, 2].axis('off')
                
                axes[0, 3].imshow(target_tensor[2], cmap='Blues', vmin=0, vmax=1)
                axes[0, 3].set_title('Target (Blue)')
                axes[0, 3].axis('off')

                # --- Row 2: Reconstructed ---
                # Reconstructed RGB
                recon_img = reconstructed_images[i].cpu().float().detach().permute(1, 2, 0).numpy()
                axes[1, 0].imshow(recon_img)
                axes[1, 0].set_title('Recon (RGB)')
                axes[1, 0].axis('off')

                # Reconstructed R/G/B Channels
                recon_tensor = reconstructed_images[i].cpu().float().detach()
                axes[1, 1].imshow(recon_tensor[0], cmap='Reds', vmin=0, vmax=1)
                axes[1, 1].set_title('Recon (Red)')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(recon_tensor[1], cmap='Greens', vmin=0, vmax=1)
                axes[1, 2].set_title('Recon (Green)')
                axes[1, 2].axis('off')
                
                axes[1, 3].imshow(recon_tensor[2], cmap='Blues', vmin=0, vmax=1)
                axes[1, 3].set_title('Recon (Blue)')
                axes[1, 3].axis('off')

                # Save figure
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f'reconstruction_8subplots_{i}.png'), 
                    dpi=150, 
                    bbox_inches='tight'
                )
                plt.close()

                # Optional: Save individual images
                save_image(next_images[i], os.path.join(save_dir, f'target_{i}.png'))
                save_image(reconstructed_images[i], os.path.join(save_dir, f'reconstructed_{i}.png'))
                
            print(f"Visualization saved to {save_dir}")
        
        
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        gen_discret_action: Optional[bool] = None,
        ar_infer: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({'gen_discret_action': gen_discret_action})
        model_inputs.update({'ar_infer': ar_infer})

        if "x" in kwargs:
            model_inputs.update({'x': kwargs['x']})
        if "proprio" in kwargs:
            model_inputs.update({'proprio': kwargs['proprio']})
        if "t" in kwargs:
            model_inputs.update({'t': kwargs['t']})

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "images": images,
                "point_cloud": point_cloud,
                "proprio": proprio,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )
        
        print(model_inputs)
        input()

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text