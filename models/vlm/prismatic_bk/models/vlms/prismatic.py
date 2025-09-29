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

import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.vlm.prismatic.models.backbones.llm import LLMBackbone
from models.vlm.prismatic.models.backbones.llm.prompting import PromptBuilder
from models.vlm.prismatic.models.backbones.vision import VisionBackbone
from models.vlm.prismatic.models.vlms.base_vlm import VLM
from models.vlm.prismatic.overwatch import initialize_overwatch
from models.vlm.prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from models.vlm.prismatic.models.image.vision_tokenizer import VisionTokenizer
from models.vlm.prismatic.models.image.vision_tokenizer import MLP_GELU
from models.vlm.prismatic.models.fuser.camera import get_camera_params, get_projection_func
from models.vlm.prismatic.models.pointcloud.backbone.pointvit import PointTokenizer
from models.vlm.prismatic.models.generation.models import MultimodalGenerationManager
from models.vlm.prismatic.models.generation.utils import images_to_patches, create_roi_mask_from_indices
from models.vlm.prismatic.models.generation.gen_loss import chamfer_distance_l2, earth_movers_distance
from models.vlm.prismatic.models.generation.visualize import visualize_generation_simple

from models.action_model import ActionEmbedder, TimestepEmbedder, LabelEmbedder, FinalLayer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)
import matplotlib.pyplot as plt

def save_projection_visualization(
    xyz_3d, 
    patch_indices, 
    valid_mask, 
    image_size=(672, 672), 
    total_stride=42,
    save_dir="projection_visualization",
    rgb_image=None,  
    background_alpha=0.4,
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
        # === Contrastive Parameters ===
        use_pointcloud: bool = False,
        use_tactile: bool = False,
        use_contrastive: bool = False,
        llm_vision_layers: int = 1,
        # === Generation Parameters ===
        use_generation: bool = True,
        gen_image: bool = False,
        num_image_gen_queries: int = 128,
        image_decoder_layers: int = 3,
        image_decoder_heads: int = 8,
        image_patch_size: int = 42, 
        use_roi: bool = False,
        roi_dilation_kernel_size: int = 3, 
        gen_pointcloud: bool = True,
        gen_tactile: bool = True,
        pointcloud_trans_dim: int = 1024,
        pointcloud_decoder_layers: int = 4,
        pointcloud_decoder_heads: int = 8,
        pointcloud_group_size: int = 8,
        pointcloud_num_groups: int = 128,
        tactile_decoder_layers: int = 2,
        tactile_decoder_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )
        self.token_size = token_size
        self.use_diff = use_diff
        self.use_pointcloud = use_pointcloud
        self.use_tactile = use_tactile
        self.use_contrastive = use_contrastive
        self.llm_vision_layers = llm_vision_layers
        
        # === Generation Configuration ===
        self.use_generation = use_generation
        self.gen_image = gen_image and use_generation
        self.use_roi = use_roi
        self.gen_pointcloud = gen_pointcloud and use_generation
        self.gen_tactile = gen_tactile and use_generation

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
        
        self.image_hidden_dim = 1024 
        self.vision_tower_2d = VisionTokenizer(input_size=self.image_hidden_dim)
        self.projector_2d = MLP_GELU(self.image_hidden_dim, token_size, 2)

        if self.use_pointcloud:
            self.vision_tower_3d = PointTokenizer(in_channels=3,
                                            embed_dim=768,
                                            depth=12,
                                            num_heads=12,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            base_ckpt_path="/media/liuzhuoyang/new_vla/Any2Point/Any2Point_CLIP_Lang/ckpts/ViT-L-14.pt",
                                        )
            self.projector_3d = MLPProjector(self.vision_tower_3d.embed_dim, token_size)
        
        if self.use_tactile:
            self.tactile_dim = 12
            self.tactile_embedder = ActionEmbedder(action_size=self.tactile_dim, hidden_size=token_size)

        # === Diffusion Components ===
        self.proprio_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)
        if self.use_diff:
            self.x_embedder = ActionEmbedder(action_size=action_dim, hidden_size=token_size)
            self.t_embedder = TimestepEmbedder(token_size)
            self.z_embedder = LabelEmbedder(in_size=token_size, hidden_size=token_size, dropout_prob=self.class_dropout_prob)
            self.final_layer = FinalLayer(token_size, action_dim)
            
        # === Generation Components ===
        if self.use_generation:
            self.generation_manager = MultimodalGenerationManager(
                token_size=token_size,
                # Image generation parameters
                use_image_generation=self.gen_image,
                num_image_gen_queries=num_image_gen_queries,
                image_decoder_layers=image_decoder_layers,
                image_decoder_heads=image_decoder_heads,
                image_patch_size=image_patch_size,
                use_roi=use_roi,
                roi_dilation_kernel_size=roi_dilation_kernel_size,
                # Point cloud generation parameters
                use_pointcloud_generation=self.gen_pointcloud,
                pointcloud_trans_dim=pointcloud_trans_dim,
                pointcloud_decoder_layers=pointcloud_decoder_layers,
                pointcloud_decoder_heads=pointcloud_decoder_heads,
                pointcloud_group_size=pointcloud_group_size,
                pointcloud_num_groups=pointcloud_num_groups,
                # Tactile generation parameters
                use_tactile_generation=self.gen_tactile,
                tactile_dim=self.tactile_dim,
                tactile_decoder_layers=tactile_decoder_layers,
                tactile_decoder_heads=tactile_decoder_heads,
            )

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_tower_2d","projector_2d",
                                "llm_backbone", "proprio_embedder"]
        if self.use_diff:
            self.all_module_keys.extend(["x_embedder", "t_embedder", "final_layer",])
        if self.use_pointcloud:
            self.all_module_keys.extend(["vision_tower_3d", "projector_3d",])
        if self.use_tactile:
            self.all_module_keys.extend(["tactile_embedder"])
        if self.use_generation:
            self.all_module_keys.append("generation_manager")
        self.trainable_module_keys = []

        self.initialize_weights()
        if self.use_pointcloud:
            self.vision_tower_3d.initialize_weights()
    
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
        
        # Initialize diffusion components if used
        if self.use_diff:
            nn.init.normal_(self.x_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.x_embedder.mlp.fc2.weight, std=0.02)
            nn.init.normal_(self.proprio_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.proprio_embedder.mlp.fc2.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            nn.init.constant_(self.final_layer.mlp.fc2.weight, 0)
            nn.init.constant_(self.final_layer.mlp.fc2.bias, 0)
    
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
            print(f"! Missed ({len(missing_keys)} parameters!):\n  {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"! Unexpected ({len(unexpected_keys)} parameters!):\n  {unexpected_keys[:3]}...")

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
        action_dim: int = 7,
        use_diff: bool = False,
        use_pointcloud: bool = False,
        use_tactile: bool = False,
        use_generation: bool = False,
        gen_image: bool = False,
        gen_pointcloud: bool = False,
        gen_tactile: bool = False,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            class_dropout_prob=class_dropout_prob,
            action_dim=action_dim,
            use_diff=use_diff,
            use_pointcloud=use_pointcloud,
            use_tactile=use_tactile,
            use_generation=use_generation,
            gen_image=gen_image,
            gen_pointcloud=gen_pointcloud,
            gen_tactile=gen_tactile,
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
        if use_pointcloud:
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

        :param stage: Pretraining stage in < "pretrain" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """

        if stage in {"pretrain"}:
            self.vision_tower_2d.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector_2d.requires_grad_(True)
            if self.use_pointcloud:
                self.vision_tower_3d.requires_grad_(True)
                self.projector_3d.requires_grad_(True)
            if self.use_tactile:
                self.tactile_embedder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_tower_2d","projector_2d",
                                          "llm_backbone", "proprio_embedder"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.use_pointcloud:
                self.trainable_module_keys.extend(["vision_tower_3d", "projector_3d"])
            if self.use_tactile:
                self.trainable_module_keys.append("tactile_embedder")

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Tower 2D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 2D ", ctx_level=1)
            if self.use_pointcloud:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Tower 3D ", ctx_level=1)
                overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 3D ", ctx_level=1)
            if self.use_tactile:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Tactile Embedder ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)

        elif stage in {"finetune"}:
            self.vision_tower_2d.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector_2d.requires_grad_(True)
            if self.use_pointcloud:
                self.vision_tower_3d.requires_grad_(False)
                self.projector_3d.requires_grad_(True)
            if self.use_tactile:
                self.tactile_embedder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone","projector_2d",
                                          "proprio_embedder"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder","final_layer"])
            if self.use_pointcloud:
                self.trainable_module_keys.extend(["projector_3d"])
            if self.use_tactile:
                self.trainable_module_keys.append("tactile_embedder")

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 2D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 2D ", ctx_level=1)
            if self.use_pointcloud:
                overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 3D ", ctx_level=1)
                overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 3D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1) 
            if self.use_tactile:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Tactile Embedder ", ctx_level=1)

        elif stage in {"post-training"}:
            self.vision_tower_2d.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector_2d.requires_grad_(True)
            if self.use_pointcloud:
                self.vision_tower_3d.requires_grad_(False)
                self.projector_3d.requires_grad_(True)
            if self.use_tactile:
                self.tactile_embedder.requires_grad_(True)
            self.generation_manager.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_tower_2d","projector_2d",
                                          "llm_backbone", "proprio_embedder"]
            if self.use_diff:
                self.trainable_module_keys.extend(["x_embedder", "t_embedder", "final_layer"])
            if self.use_pointcloud:
                self.trainable_module_keys.extend(["vision_tower_3d", "projector_3d"])
            if self.use_tactile:
                self.trainable_module_keys.append("tactile_embedder")
            self.trainable_module_keys.append("generation_manager")

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 2D ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 2D ", ctx_level=1)
            if self.use_pointcloud:
                overwatch.info(f"[Frozen]    🥶 =>> Vision Tower 3D ", ctx_level=1)
                overwatch.info(f"[TRAINABLE] 🔥 =>> Projector 3D ", ctx_level=1)
            if self.use_tactile:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Tactile Embedder ", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Generation Manager ", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported! Try < pretrain | finetune | post-training >")

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
            module_classes={PointTokenizer, VisionTokenizer},
        )
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={MLP_GELU, MultimodalGenerationManager},
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
        self, images, pointcloud, tactile, gripper_xyz, camera_name
    ):
        if isinstance(images, dict):
            assert 'front_image' in images, "front_image must be present in multi-view images"
            
            front_camera_params = get_camera_params(camera_name, device=images['front_image'].device if pointcloud is None else pointcloud.device)
            projected_front_tokens, patch_hw = self.encode_images(images['front_image'])
            projected_front_tokens = torch.stack(projected_front_tokens, dim=0)
            
            if self.use_pointcloud and pointcloud is not None:
                pointcloud_patch_embeddings, pointcloud_centers = self.vision_tower_3d(pointcloud)
                projected_pointcloud_tokens = self.projector_3d(pointcloud_patch_embeddings)
                project_func = get_projection_func(camera_name)
                patch_indices, valid_mask = project_func(
                    pointcloud_centers, 
                    front_camera_params.K,
                    front_camera_params.R,
                    front_camera_params.t,
                    image_size_resize=(672, 672),
                    vision_strides={'patch_stride': 14, 'conv_stride': 3}
                )
            else:
                B = projected_front_tokens.shape[0]  
                N_pc = projected_front_tokens.shape[1]  
                projected_pointcloud_tokens = torch.zeros(
                    (B, N_pc, self.token_size), 
                    dtype=projected_front_tokens.dtype,
                    device=projected_front_tokens.device
                )
                patch_indices = torch.zeros(
                    (B, N_pc, 2), 
                    dtype=torch.long, 
                    device=projected_front_tokens.device
                )
                valid_mask = torch.zeros(
                    (B, N_pc), 
                    dtype=torch.bool, 
                    device=projected_front_tokens.device
                )
                
            ## project visualization
            # save_projection_visualization(
            #     pointcloud_centers, 
            #     patch_indices, 
            #     valid_mask, 
            #     save_dir="/media/liuzhuoyang/new_vla/Rec_Tac_Diff_beta/LLM_policy/vis", 
            #     rgb_image=images['front_image'],
            # )
            # input("Press Enter to continue...")

            assert projected_pointcloud_tokens.shape[1] == projected_front_tokens.shape[1], \
                f"Token count mismatch: PC={projected_pointcloud_tokens.shape[1]}, Front Img={projected_front_tokens.shape[1]}"
            
            fused_tokens = torch.cat(
                [projected_pointcloud_tokens, projected_front_tokens], 
                dim=1
            )
            
            other_views = [key for key in images.keys() if key != 'front_image']
            for view_key in other_views:
                projected_view_tokens, _ = self.encode_images(images[view_key])
                projected_view_tokens = torch.stack(projected_view_tokens, dim=0)
                fused_tokens = torch.cat([fused_tokens, projected_view_tokens], dim=1)
            
        else:
            camera_params = get_camera_params(camera_name, device=images.device if pointcloud is None else pointcloud.device)
            # 2D tokens
            projected_image_tokens, patch_hw = self.encode_images(images)
            projected_image_tokens = torch.stack(projected_image_tokens, dim=0)

            # 3D tokens
            if self.use_pointcloud and pointcloud is not None:
                pointcloud_patch_embeddings, pointcloud_centers = self.vision_tower_3d(pointcloud)
                projected_pointcloud_tokens = self.projector_3d(pointcloud_patch_embeddings)
                project_func = get_projection_func(camera_name)
                patch_indices, valid_mask = project_func(
                    pointcloud_centers, 
                    camera_params.K,
                    camera_params.R,
                    camera_params.t,
                    image_size_resize=(672, 672),
                    vision_strides={'patch_stride': 14, 'conv_stride': 3}
                )
            else:
                B = projected_image_tokens.shape[0]  
                N_pc = projected_image_tokens.shape[1]  
                projected_pointcloud_tokens = torch.zeros(
                    (B, N_pc, self.token_size), 
                    dtype=projected_image_tokens.dtype,
                    device=projected_image_tokens.device
                )
                patch_indices = torch.zeros(
                    (B, N_pc, 2), 
                    dtype=torch.long, 
                    device=projected_image_tokens.device
                )
                valid_mask = torch.zeros(
                    (B, N_pc), 
                    dtype=torch.bool, 
                    device=projected_image_tokens.device
                )

            assert projected_pointcloud_tokens.shape[1] == projected_image_tokens.shape[1], \
                f"Token count mismatch: PC={projected_pointcloud_tokens.shape[1]}, Img={projected_image_tokens.shape[1]}"
            
            fused_tokens = torch.cat(
                [projected_pointcloud_tokens, projected_image_tokens], 
                dim=1
            )

        positive_pc_indices_for_tac = None
        linear_positive_img_indices_for_tac = None
        pointcloud_centers_for_tac = None
        if self.use_tactile and tactile is not None:
            last_dim = gripper_xyz.shape[-1]
            if last_dim % 3 != 0:
                raise ValueError(f"gripper_xyz last dimension ({last_dim}) is not divisible by 3")
            n_arms = last_dim // 3

            # get tactile tokens
            t = tactile
            t_flat = t.view(t.shape[0], -1)
            if t_flat.shape[-1] == self.tactile_dim * n_arms:
                tactile_slices = torch.chunk(t_flat, n_arms, dim=-1)
            else:
                raise ValueError(
                    f"Unexpected tactile shape {tuple(t.shape)}. "
                    f"Expect either (B, {self.tactile_dim * n_arms}) or (B, {n_arms}, {per_arm_dim})."
                )
            
            tac_emb_list = []
            for ts in tactile_slices:
                emb = self.tactile_embedder(ts) 
                tac_emb_list.append(emb.unsqueeze(1))
            tac_embedding = torch.cat(tac_emb_list, dim=1) # [B, n_arms, token_size]

            fused_tokens = torch.cat(
                [fused_tokens, tac_embedding],
                dim=1
            )
            B = tac_embedding.shape[0]
            gripper_xyz_expanded = gripper_xyz.view(B, n_arms, 3)

            distances = torch.cdist(gripper_xyz_expanded, pointcloud_centers)  
            _, positive_pc_indices_for_tac = torch.topk(distances, k=1, dim=2, largest=False)
            patch_w = int(projected_front_tokens.shape[1]**0.5) # 16
            expanded_indices_for_gather = positive_pc_indices_for_tac.unsqueeze(-1).expand(-1, -1, -1, 2)  # [B, n_arms, 1, 2]
            positive_img_indices_2d = torch.gather(patch_indices.unsqueeze(1).expand(-1, n_arms, -1, -1), 2, expanded_indices_for_gather) # [B, n_arms, 1, 2]
            linear_positive_img_indices_for_tac = (
                positive_img_indices_2d[..., 0] * patch_w + positive_img_indices_2d[..., 1]
            ) # [B, n_arms, 1]
            pointcloud_centers_for_tac = pointcloud_centers # [32, 256, 3]

        else:
            B = fused_tokens.shape[0]  
            N_tac = 1  
            tac_embedding = torch.zeros(
                    (B, N_tac, self.token_size), 
                    dtype=fused_tokens.dtype,
                    device=fused_tokens.device
                )
            fused_tokens = torch.cat(
                [fused_tokens, tac_embedding],
                dim=1
            )

        # print(fused_tokens.shape)
        # input()
            
        return (fused_tokens, patch_indices, valid_mask, 
            positive_pc_indices_for_tac, linear_positive_img_indices_for_tac, pointcloud_centers_for_tac)
    
    def compute_generation_losses(
        self, 
        generation_outputs: Dict[str, torch.Tensor], 
        next_images: Optional[torch.Tensor] = None, 
        next_point_cloud: Optional[torch.Tensor] = None,
        next_tactile: Optional[torch.Tensor] = None,
    ):
        losses = {}
        total_loss = 0.0
        if self.gen_image and next_images is not None and 'image_generation' in generation_outputs:
            image_gen_loss_total = 0.0
            generated_patches = generation_outputs['image_generation']  # [B, 256, patch_dim]
            generation_roi_mask = generation_outputs['generation_roi_mask']  # [B, 256] bool
            next_images_patches = images_to_patches(next_images, self.generation_manager.image_gen_module.image_patch_size)  # [B, 256, patch_dim]

            pred_roi = generated_patches[generation_roi_mask]
            gt_roi = next_images_patches[generation_roi_mask]

            # main generation loss (hybrid)
            if pred_roi.numel() > 0:
                gen_mse = F.mse_loss(pred_roi, gt_roi)
                gen_l1 = F.l1_loss(pred_roi, gt_roi)
                image_gen_loss = gen_mse + 0.5 * gen_l1
                losses['image_roi_generation_loss'] = image_gen_loss
                total_loss = total_loss + image_gen_loss
                image_gen_loss_total = image_gen_loss_total + image_gen_loss

            # background consistency loss
            bg_mask = ~generation_roi_mask
            pred_bg = generated_patches[bg_mask]
            gt_bg = next_images_patches[bg_mask]
            if pred_bg.numel() > 0:
                bg_l1 = F.l1_loss(pred_bg, gt_bg)
                losses['bg_consistency_loss'] = 0.01 * bg_l1
                total_loss = total_loss + losses['bg_consistency_loss']
                image_gen_loss_total = image_gen_loss_total + losses['bg_consistency_loss']
            
            # delta regularization loss
            if 'delta_all' in generation_outputs:
                # generation_outputs['delta_all']: [B, num_patches, patch_dim]
                delta_norm = generation_outputs['delta_all'].abs().mean()
                delta_loss = -0.1 * delta_norm  
                losses['delta_magnitude_reward'] = delta_loss
                total_loss = total_loss + delta_loss
                image_gen_loss_total = image_gen_loss_total + delta_loss
            losses['image_gen_loss'] = image_gen_loss_total

        if self.gen_pointcloud and next_point_cloud is not None and 'pointcloud_coord_generation' in generation_outputs:
            # Extract coordinates from ground truth
            assert next_point_cloud.shape[2] == 3, "Point cloud must have 3 dimensions (XYZ)"

            pc_coord_pred = generation_outputs['pointcloud_coord_generation']
            pc_coord_loss = chamfer_distance_l2(pc_coord_pred, next_point_cloud) 
            losses['point_cloud_gen_loss'] = pc_coord_loss
            total_loss += pc_coord_loss

        if self.gen_tactile and next_tactile is not None and 'tactile_generation' in generation_outputs:
            tactile_pred = generation_outputs['tactile_generation']  # (B, tactile_dim)
            gt_tactile = next_tactile  # (B, tactile_dim)
            # print("tactile_pred:", tactile_pred, "gt_tactile:", gt_tactile)
            # input()
            indices_to_keep = list(range(4)) + list(range(6, 10)) 
            tactile_pred_important = tactile_pred[:, indices_to_keep]
            gt_tactile_important = gt_tactile[:, indices_to_keep]
            
            # print("tactile_pred_important:", tactile_pred_important[0], "gt_tactile_important:", gt_tactile_important[0])
            # input()
            
            tactile_loss = F.mse_loss(tactile_pred_important, gt_tactile_important) 
            losses['tactile_gen_loss'] = tactile_loss
            total_loss = total_loss + tactile_loss

        losses['total_generation_loss'] = total_loss
        return losses
    
    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        z: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        gripper_xyz: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        camera_name: Optional[str] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        tactile: Optional[torch.FloatTensor] = None,
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
        next_images: Optional[torch.FloatTensor] = None,
        next_point_cloud: Optional[torch.FloatTensor] = None,
        next_tactile: Optional[torch.FloatTensor] = None,
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
        projected_fused_embeddings, patch_indices, valid_mask, \
            positive_pc_indices_for_tac, linear_positive_img_indices_for_tac, \
            pointcloud_centers_for_tac = self.get_fused_tokens(
                images, point_cloud, tactile, gripper_xyz, camera_name
            )
        N_pc = 256
        N_img = 256
        bos_token_len = 1
        pc_tokens_start_idx = bos_token_len
        pc_tokens_end_idx = pc_tokens_start_idx + N_pc
        img_tokens_start_idx = pc_tokens_end_idx
        img_tokens_end_idx = img_tokens_start_idx + N_img
        current_front_image_features = projected_fused_embeddings[:, N_pc: N_pc+N_img, :]
        
        if self.use_tactile:
            N_tac = self.action_dim // 7
            tac_tokens_start_idx = img_tokens_end_idx
            tac_tokens_end_idx = tac_tokens_start_idx + N_tac
        
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        
        # Create initial embeddings with fused tokens
        z = torch.cat([input_embeddings[:, :1, :], 
                       projected_fused_embeddings, 
                       input_embeddings[:, 1:, :]], dim=1
                    )

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
            tac_token_indices=(tac_tokens_start_idx, tac_tokens_end_idx) if self.use_tactile else None,
            patch_correspondence_indices=patch_indices,
            correspondence_valid_mask=valid_mask,
            positive_pc_indices_for_tac=positive_pc_indices_for_tac,
            linear_positive_img_indices_for_tac=linear_positive_img_indices_for_tac,
            compute_token_contrastive_loss=self.use_contrastive, 
            compute_tactile_contrastive_loss=(self.use_contrastive and self.use_tactile),
        )
        
        # === Generation Forward Pass ===
        generation_outputs = {}
        generation_losses = {}
        
        if self.use_generation and (self.gen_image or self.gen_pointcloud or self.gen_tactile) and self.training and output.hidden_states is not None:
            # Use the last layer hidden states for generation
            llm_hidden_states = output.hidden_states[-1]  # [B, seq_len, hidden_dim]
            current_images_patches = None
            roi_mask_2d = torch.ones(
                patch_indices.shape[0],
                16, 
                16, 
                dtype=torch.bool, 
                device=patch_indices.device
            )
            if self.gen_image and images is not None:
                current_images_patches = images_to_patches(images['front_image'][:, :3, :, :], 
                                                         self.generation_manager.image_gen_module.image_patch_size
                                                    )
                if self.use_roi:
                    roi_mask_2d = create_roi_mask_from_indices(patch_indices)
            
            # Perform multimodal generation
            generation_outputs = self.generation_manager(
                llm_hidden_states=llm_hidden_states,
                current_image_features=current_front_image_features,
                current_images_patches=current_images_patches,
                current_point_cloud=None,
                roi_mask_2d=roi_mask_2d,
            )
            
            if self.gen_image:
                assert next_images is not None
            if self.gen_pointcloud:
                assert next_point_cloud is not None
            if self.gen_tactile:
                assert next_tactile is not None
            generation_losses = self.compute_generation_losses(
                generation_outputs, 
                next_images=next_images, 
                next_point_cloud=next_point_cloud,
                next_tactile=next_tactile,
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
            
            # Return with generation information
            if self.training:
                visualize_generation_simple(
                    generation_outputs, 
                    next_images,
                    next_point_cloud,
                    "/media/liuzhuoyang/new_vla/Rec_Tac_Diff_beta/LLM_policy/vis/gen_vis_pretrain-0818_egg_tac"
                )
                return output, noise_pred, generation_outputs, generation_losses
            else:
                return output, noise_pred
        
        # # Return with generation information
        if self.training:
            return output, generation_outputs, generation_losses
        else:
            return output # autoregressive mode
        
        
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