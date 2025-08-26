"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy
import time

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast

from vlm.prismatic.models.backbones.llm import LLMBackbone
from vlm.prismatic.models.backbones.llm.prompting import PromptBuilder
from vlm.prismatic.models.backbones.vision import VisionBackbone
from vlm.prismatic.models.vlms.base_vlm import VLM
from vlm.prismatic.models.vlms.prismatic import PrismaticVLM
from vlm.prismatic.overwatch import initialize_overwatch
from vlm.prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from action_model import create_diffusion

from vlm.prismatic.models.image.vision_tokenizer import VisionTokenizer
from vlm.prismatic.models.image.vision_tokenizer import MLP_GELU
from vlm.prismatic.models.pointcloud.backbone.pointvit import PointViT

from action_model.action_model import ActionModel
from action_model.models import DiT

from vlm.prismatic.vla.action_tokenizer import ActionTokenizer
import json
import time
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class CogACT(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_tokenizer: ActionTokenizer,
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        use_diff: bool = False,
        use_pointcloud: bool = False,
        use_tactile: bool = False,
        use_contrastive: bool = False,
        use_reconstruction: bool = False,
        recon_image: bool = False,
        recon_pointcloud: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.action_tokenizer = action_tokenizer

        self.use_diff = use_diff
        self.use_pointcloud = use_pointcloud
        self.use_tactile = use_tactile
        self.use_contrastive = use_contrastive
        self.use_reconstruction = use_reconstruction
        self.recon_image = recon_image
        self.recon_pointcloud = recon_pointcloud
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.vlm.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.all_module_keys=[]
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)

        self.use_ema = use_ema
        self.norm_stats = norm_stats
        self._trainable_module_keys = []

        if self.use_diff:
            self.ddim_diffusion = None
            self.diffusion_steps = 100
            self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'squaredcos_cap_v2', diffusion_steps=100, sigma_small=True, learn_sigma = False)

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        next_images: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        next_point_cloud: Optional[torch.FloatTensor] = None,
        tactile_right: Optional[torch.FloatTensor] = None,
        tactile_left: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        gripper_xyz: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 3,
        action_masks = None,
        use_diff: Optional[bool] = None,
        stage: str = "",
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        if use_diff is not None:
            self.use_diff = use_diff

        if self.use_diff:
            proprio = proprio.repeat(repeated_diffusion_steps, *([1] * (proprio.ndimension() - 1)))
            gripper_xyz = gripper_xyz.repeat(repeated_diffusion_steps, *([1] * (gripper_xyz.ndimension() - 1)))
            
            actions = actions.repeat(repeated_diffusion_steps, *([1] * (actions.ndimension() - 1)))
            actions_history = actions[:,0: self.past_action_window_size,:]
            actions_future = actions[:, -(self.future_action_window_size+1):, :]

            input_ids = input_ids.repeat(repeated_diffusion_steps, *([1] * (input_ids.ndimension() - 1)))
            attention_mask = attention_mask.repeat(repeated_diffusion_steps, *([1] * (attention_mask.ndimension() - 1)))
            action_masks = action_masks.repeat(repeated_diffusion_steps, *([1] * (action_masks.ndimension() - 1)))
            labels = labels.repeat(repeated_diffusion_steps, *([1] * (labels.ndimension() - 1)))

            if isinstance(images, dict):
                images = {
                    k: v.repeat(repeated_diffusion_steps, *([1] * (v.ndimension() - 1)))
                    for k, v in images.items()
                }
            else:
                images = images.repeat(repeated_diffusion_steps, *([1] * (images.ndimension() - 1)))
            if self.use_reconstruction and self.recon_image:
                next_images = next_images.repeat(repeated_diffusion_steps, *([1] * (next_images.ndimension() - 1)))
            if self.use_pointcloud:
                point_cloud = point_cloud.repeat(repeated_diffusion_steps, *([1] * (point_cloud.ndimension() - 1)))
            if self.use_tactile:
                tactile_right = tactile_right.repeat(repeated_diffusion_steps, *([1] * (tactile_right.ndimension() - 1)))
                tactile_left = tactile_left.repeat(repeated_diffusion_steps, *([1] * (tactile_left.ndimension() - 1)))
            if self.use_pointcloud and self.use_reconstruction and self.recon_pointcloud:
                next_point_cloud = next_point_cloud.repeat(repeated_diffusion_steps, *([1] * (next_point_cloud.ndimension() - 1)))
        
            noise = torch.randn_like(actions_future)  # [B, T, C]
            timestep = torch.randint(0, self.diffusion.num_timesteps, (actions_future.size(0),), device= actions.device)
            x = self.diffusion.q_sample(actions_future, timestep, noise)

            output, noise_pred, reconstruction_outputs, reconstruction_losses = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                next_images=next_images,
                point_cloud=point_cloud,
                next_point_cloud=next_point_cloud,
                tactile_right=tactile_right,
                tactile_left=tactile_left,
                labels=labels,
                x=x,
                t=timestep,
                proprio=proprio,
                gripper_xyz=gripper_xyz,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_diff = self.use_diff,
            )
            assert noise_pred.shape == noise.shape == actions.shape
            loss_dict = {
                'total_loss': torch.tensor(0, dtype=torch.float32) ,
                'contrastive_loss': torch.tensor(0, dtype=torch.float32) ,
                'diff_loss': torch.tensor(0, dtype=torch.float32) ,
                'image_recon_loss': torch.tensor(0, dtype=torch.float32) ,
                'point_cloud_recon_loss': torch.tensor(0, dtype=torch.float32) ,
            }
            diff_loss = ((noise_pred - noise) ** 2).mean()
            loss_dict['total_loss'] = diff_loss
            loss_dict['diff_loss'] = diff_loss
            if self.use_reconstruction and self.recon_image:
                loss_dict['image_recon_loss'] = reconstruction_losses['image_recon_loss']
                loss_dict['total_loss'] += reconstruction_losses['image_recon_loss']
            if self.use_reconstruction and self.recon_pointcloud:
                loss_dict['point_cloud_recon_loss'] = reconstruction_losses['point_cloud_recon_loss']
                loss_dict['total_loss'] += reconstruction_losses['point_cloud_recon_loss']
            if self.use_contrastive:
                loss_dict['contrastive_loss'] = output.contrastive_loss
                loss_dict['total_loss'] += output.contrastive_loss
                if self.use_tactile:
                    loss_dict['total_loss'] += output.tactile_contrastive_loss
            return loss_dict, output
        else:
            output = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images = images,
                point_cloud = point_cloud,
                labels=labels,
                proprio=proprio,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_diff = self.use_diff,
            )
            if self.use_reconstruction:
                output.loss += reconstruction_losses['total_reconstruction_loss']
            return output
        

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={PointViT, VisionTokenizer},
        )
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={MLPProjector, MLP_GELU},
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

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        action_tokenizer: ActionTokenizer,
        pretrained_checkpoint: Path,
        model_id: str,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats = None,
        class_dropout_prob: float = 0.0,
        use_diff: bool = False,
        use_pointcloud: bool = False,
        use_tactile: bool = False,
        use_contrastive: bool = False,
        use_reconstruction: bool = False,
        recon_image: bool = False,
        recon_pointcloud: bool = False,
        **kwargs,
    ) -> CogACT:

        # Load VLM backbone, borrowed from PrismaticVLM
        # 29.1G
        vlm = PrismaticVLM(
            model_id,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            class_dropout_prob=class_dropout_prob,
            use_diff=use_diff,
            action_dim=action_dim,
            use_pointcloud=use_pointcloud,
            use_tactile=use_tactile,
            use_contrastive=use_contrastive,
            use_reconstruction=use_reconstruction,
            recon_image=recon_image,
            recon_pointcloud=recon_pointcloud,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
            
        # From 2D Pretrain
        if "vision_tower_2d" in model_state_dict.keys():
            vlm.vision_tower_2d.load_state_dict(model_state_dict["vision_tower_2d"])
            print("\n\nSuccessfully loaded vision_tower_2d !!!!\n")
        else:
            print("\n\nNo vision_tower_2d found in checkpoint, initializing a new one!!!!\n")
            
        # Try to load projector_2d
        if "projector_2d" in model_state_dict.keys():
            vlm.projector_2d.load_state_dict(model_state_dict["projector_2d"])
            print("\n\nSuccessfully loaded projector_2d !!!!\n")
        else:
            print("\n\nNo projector_2d found in checkpoint, initializing a new one!!!!\n")

        # Try to load vision_tower_3d
        if use_pointcloud and "vision_tower_3d" in model_state_dict.keys():
            vlm.vision_tower_3d.load_state_dict(model_state_dict["vision_tower_3d"])
            print("\n\nSuccessfully loaded vision_tower_3d !!!!\n")
        else:
            # vlm.load_encoder_to_vision_tower(ckpt_path="/media/liuzhuoyang/new_vla/Any2Point/Any2Point_CLIP_Lang/ckpts/Language_CLIP_Scan.pth",
            #     vision_tower_3d=vlm.vision_tower_3d)
            print("\n\nNo vision_tower_3d found in checkpoint, load from Any2Point ckpt!!!!\n")
            
        # Try to load projector_3d
        if use_pointcloud and "projector_3d" in model_state_dict.keys():
            vlm.projector_3d.load_state_dict(model_state_dict["projector_3d"])
            print("\n\nSuccessfully loaded projector_3d !!!!\n")
        else:
            print("\n\nNo projector_3d found in checkpoint, initializing a new one!!!!\n")

        # Load LLM backbone
        assert ("llm_backbone" in model_state_dict
                ),"PrismaticVLM `from_pretrained` expects checkpoint with keys for `llm_backbone`!"
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"], strict=False)
        print("\n\nSuccessfully loaded llm_backbone!!!!\n")

        # Try to load proprio_embedder
        if "proprio_embedder" in model_state_dict.keys():
            vlm.proprio_embedder.load_state_dict(model_state_dict["proprio_embedder"])
            print("\n\nSuccessfully loaded proprio_embedder!!!!\n")
        else:
            print("\n\nNo proprio_embedder found in checkpoint, initializing a new one!!\n")
            
        if "tactile_embedder" in model_state_dict.keys():
            vlm.tactile_embedder.load_state_dict(model_state_dict["tactile_embedder"])
            print("\n\nSuccessfully loaded tactile_embedder!!!!\n")
        else:
            print("\n\nNo tactile_embedder found in checkpoint, initializing a new one!!\n")
            
        if use_diff and "x_embedder" in model_state_dict.keys() and "t_embedder" in model_state_dict.keys() and "final_layer" in model_state_dict.keys() and use_diff:
            vlm.x_embedder.load_state_dict(model_state_dict["x_embedder"])
            vlm.t_embedder.load_state_dict(model_state_dict["t_embedder"])
            vlm.final_layer.load_state_dict(model_state_dict["final_layer"])
            print("\n\nSuccessfully loaded x_embedder, proprio_embedder, t_embedder, final_layer from checkpoint!!!!\n")
        else:
            print("\n\nNo x_embedder, t_embedder, final_layer found in checkpoint!!!!\n")

        if use_reconstruction and "reconstruction_manager" in model_state_dict.keys() and recon_image:
            recon_state_dict = {}
            for k, v in model_state_dict["reconstruction_manager"].items():
                if k.startswith("image_recon_module."):
                    new_key = k[len("image_recon_module."):]
                    recon_state_dict[new_key] = v
            if recon_state_dict:
                vlm.reconstruction_manager.image_recon_module.load_state_dict(recon_state_dict)
                print("\n\nSuccessfully loaded reconstruction_manager.image_recon_module from checkpoint!!!!\n")
            else:
                print("\n\nNo reconstruction_manager.image_recon_module found in checkpoint!!!!\n")
        else:
            print("\n\nNo reconstruction_manager.image_recon_module found in checkpoint!!!!\n")

        if use_reconstruction and "reconstruction_manager" in model_state_dict.keys() and recon_pointcloud:
            recon_state_dict = {}
            for k, v in model_state_dict["reconstruction_manager"].items():
                if k.startswith("pointcloud_recon_module."):
                    new_key = k[len("pointcloud_recon_module."):]
                    recon_state_dict[new_key] = v
            if recon_state_dict:
                vlm.reconstruction_manager.pointcloud_recon_module.load_state_dict(recon_state_dict)
                print("\n\nSuccessfully loaded reconstruction_manager.pointcloud_recon_module from checkpoint!!!!\n")
            else:
                print("\n\nNo reconstruction_manager.pointcloud_recon_module found in checkpoint!!!!\n")
        else:
            print("\n\nNo reconstruction_manager.image_recon_module found in checkpoint!!!!\n")

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize CogACT
        cogact = CogACT(vlm,
                        action_tokenizer,
                        token_size=vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim=action_dim,
                        future_action_window_size=future_action_window_size,
                        past_action_window_size=past_action_window_size,
                        action_model_type=action_model_type,
                        use_ema=use_ema,
                        norm_stats=norm_stats,
                        use_diff=use_diff,
                        use_pointcloud=use_pointcloud,
                        use_tactile=use_tactile,
                        use_contrastive=use_contrastive,
                        use_reconstruction=use_reconstruction,
                        recon_image=recon_image,
                        recon_pointcloud=recon_pointcloud,
                    )

        return cogact        

    @torch.inference_mode()
    def predict_action_ar(
        self, 
        image: Optional[Image] = None,  # 默认值为None
        pointcloud : Optional[torch.FloatTensor] = None,
        instruction: Optional[str] = None,
        unnorm_key: Optional[str] = None, 
        cur_robot_state: Optional[str] = None,
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        action_dim: int = 7,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.get_vision_tower_2d().image_processor, self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        if cur_robot_state is not None:
            proprio_norm_stats = self.get_proprio_stats(unnorm_key)
            mask = proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["q01"], dtype=bool))
            proprio_high, proprio_low = np.array(proprio_norm_stats["q99"]), np.array(proprio_norm_stats["q01"])
            cur_robot_state = np.where(
                mask,
                2 * (cur_robot_state - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                cur_robot_state,
            )
            cur_robot_state = np.clip(cur_robot_state, -1, 1)
            cur_robot_state = torch.tensor(cur_robot_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.vlm.device)

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.vlm.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        image = image_transform.preprocess(image, return_tensors='pt')['pixel_values'][0]   
        image = image.unsqueeze(0).to(self.vlm.device)  # (1, 3, 672, 672)

        # Preprocess PointCloud
        pointcloud = pointcloud.to(self.vlm.device).contiguous()
        
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                                        # Shape: [1, seq]
                image = image,   
                point_cloud = pointcloud,                                    # Shape: [1, N, 3]
                proprio=cur_robot_state,
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1) 

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions
    

    @torch.inference_mode()
    def predict_action_diff(
        self, 
        image: Optional[Image] = None,  # 默认值为None
        pointcloud: Optional[torch.FloatTensor] = None,
        instruction: Optional[str] = None,
        cur_robot_state: Optional[str] = None,
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 0.0, 
        use_ddim: bool = True,
        num_ddim_steps: int = 8,
        action_dim: int = 7,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        self.vlm.eval()
        device = self.vlm.device
        image_transform, tokenizer = self.vlm.get_vision_tower_2d().image_processor, self.vlm.llm_backbone.tokenizer
        
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Build VLA Prompt
        message = f"What action should the robot take to {instruction.lower()}?"
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=message)
        prompt_text = prompt_builder.get_prompt()
        
        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        # print(input_ids)
        # input()
        if not isinstance(tokenizer, LlamaTokenizerFast):
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")
        def append_tokens(ids_to_append):
            token_tensor = torch.tensor([ids_to_append], dtype=torch.long, device=device)
            return torch.cat((input_ids, token_tensor), dim=1)
        has_empty_token = lambda: torch.all(input_ids[:, -1] == 29871)
        if not has_empty_token():
            input_ids = append_tokens([29871, 32001, 32002, 29871])
        # print(input_ids)
        # input()
        
        # input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        # if isinstance(tokenizer, LlamaTokenizerFast):
        #     # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        #     # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        #     if not torch.all(input_ids[:, -1] == 29871):
        #         input_ids = torch.cat(
        #             (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.vlm.device)), dim=1
        #         )
        # else:
        #     raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")
        
        # Preprocess Image
        image_mask = torch.ones(1, 672, 672) 
        image = image_transform.preprocess(image, return_tensors='pt')['pixel_values'][0]   
        image = torch.cat([image, image_mask], dim=0)
        image = image.unsqueeze(0).to(self.vlm.device)
        
        # Preprocess PointCloud
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud)
        pointcloud = pointcloud.to(self.vlm.device).contiguous()
        
        # Preprocess robot state
        if cur_robot_state is not None:
            proprio_norm_stats = self.get_proprio_stats(unnorm_key)
            mask = proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["q01"], dtype=bool))
            proprio_high, proprio_low = np.array(proprio_norm_stats["q99"]), np.array(proprio_norm_stats["q01"])
            cur_robot_state = np.where(
                mask,
                2 * (cur_robot_state - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                cur_robot_state,
            )
            cur_robot_state = np.clip(cur_robot_state, -1, 1)
            cur_robot_state = torch.tensor(cur_robot_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.vlm.device)

        def unnormalize_actions(normalized_actions):
            action_norm_stats = self.get_action_stats(unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            
            normalized_actions = np.clip(normalized_actions, -1, 1)
            
            if isinstance(normalized_actions, np.ndarray):
                if normalized_actions.ndim == 1 and len(normalized_actions) == 7:
                    normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1)
                elif normalized_actions.ndim == 1 and len(normalized_actions) == 14:
                    normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1)
                    normalized_actions[13] = np.where(normalized_actions[13] < 0.5, 0, 1)
                elif normalized_actions.ndim > 1:
                    if normalized_actions.shape[1] == 7:
                        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
                    elif normalized_actions.shape[1] == 14:
                        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
                        normalized_actions[:, 13] = np.where(normalized_actions[:, 13] < 0.5, 0, 1)
            
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            return actions

        def prepare_diffusion(input_ids_diff=None):
            noise = torch.randn(1, self.future_action_window_size+1, action_dim, device=device)
            timestep = torch.randint(0, self.diffusion.num_timesteps, (self.future_action_window_size+1,), device=device)
            using_cfg = cfg_scale > 1.0
            
            if input_ids_diff is None:
                input_ids_diff = input_ids
                input_ids_diff = input_ids_diff[:, :-3] # prism-dinosiglip-224px+7b
            
            if using_cfg:
                noise = torch.cat([noise, noise], 0)
                uncondition = self.vlm.z_embedder.uncondition.unsqueeze(0).expand(input_ids_diff.shape[0], 1, -1)
                sample_fn = self.vlm.forward_with_cfg
                model_kwargs = {
                    'z': uncondition, 
                    'cfg_scale': cfg_scale, 
                    'input_ids': input_ids_diff, 
                    'image': image,
                    'point_cloud': pointcloud,
                }
                if cur_robot_state is not None:
                    model_kwargs['proprio'] = cur_robot_state
            else:
                model_kwargs = {'input_ids': input_ids_diff, 
                                'images': image,
                                'point_cloud': pointcloud,
                            }
                if cur_robot_state is not None:
                    model_kwargs['proprio'] = cur_robot_state
                sample_fn = self.vlm.forward
            
            return noise, timestep, sample_fn, model_kwargs, using_cfg
        
        def sample_diffusion(noise, sample_fn, model_kwargs, using_cfg):
            if use_ddim and num_ddim_steps is not None:
                if self.ddim_diffusion is None:
                    self.create_ddim(ddim_step=num_ddim_steps)
                samples = self.ddim_diffusion.ddim_sample_loop(
                    sample_fn, 
                    noise.shape, 
                    noise, 
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                    eta=0.0
                )
            else:
                samples = self.diffusion.p_sample_loop(
                    sample_fn, 
                    noise.shape, 
                    noise, 
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device
                )
            
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  
            
            return samples[0].cpu().numpy()
        
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            noise, timestep, sample_fn, model_kwargs, using_cfg = prepare_diffusion()
            normalized_actions = sample_diffusion(noise, sample_fn, model_kwargs, using_cfg)
        return unnormalize_actions(normalized_actions)


    @torch.inference_mode()
    def predict_action_diff_ar(
        self, 
        front_image: Optional[Image] = None,  # 默认值为None
        wrist_image: Optional[Image] = None,  # 默认值为None
        wrist_left_image :Optional[Image] = None,
        instruction: str = "", 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        action_dim: int = 7,
        cur_robot_state: Optional[str] = None,
        multi_view: bool = True,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
        # cur_robot_state = None
        message = f"What action should the robot take to {instruction.lower()}?"
        if cur_robot_state is not None:
            proprio_norm_stats = self.get_proprio_stats(unnorm_key)
            mask = proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["q01"], dtype=bool))
            proprio_high, proprio_low = np.array(proprio_norm_stats["q99"]), np.array(proprio_norm_stats["q01"])
            cur_robot_state = np.where(
                mask,
                2 * (cur_robot_state - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                cur_robot_state,
            )
            cur_robot_state = np.clip(cur_robot_state, -1, 1)
            cur_robot_state = torch.tensor(cur_robot_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.vlm.device)
            # cur_robot_state = self.action_tokenizer(cur_robot_state)
            # message = f"The current robot state is {cur_robot_state}. " + message
        
        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=message)
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)

        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871, 32001, 32002, 29871]).long(), dim=0).to(self.vlm.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = {}
        if front_image:
            front_pixel_values = image_transform(front_image)
            if isinstance(front_pixel_values, torch.Tensor):
                front_pixel_values = front_pixel_values[None, ...].to(self.vlm.device)
            elif isinstance(front_pixel_values, dict):
                front_pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in front_pixel_values.items()}
            else:
                raise ValueError(f"Unsupported `front_pixel_values` type = {type(front_pixel_values)}")
            for key, value in front_pixel_values.items():
                pixel_values[f"front_{key}"] = value
        if wrist_image:
            wrist_pixel_values = image_transform(wrist_image)
            if isinstance(wrist_pixel_values, torch.Tensor):
                wrist_pixel_values = wrist_pixel_values[None, ...].to(self.vlm.device)
            elif isinstance(wrist_pixel_values, dict):
                wrist_pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in wrist_pixel_values.items()}
            else:
                raise ValueError(f"Unsupported `wrist_pixel_values` type = {type(wrist_pixel_values)}")
            for key, value in wrist_pixel_values.items():
                pixel_values[f"wrist_{key}"] = value
        if wrist_left_image:
            wrist_left_pixel_values = image_transform(wrist_left_image)
            if isinstance(wrist_left_pixel_values, torch.Tensor):
                wrist_left_pixel_values = wrist_left_pixel_values[None, ...].to(self.vlm.device)
            elif isinstance(wrist_left_pixel_values, dict):
                wrist_left_pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in wrist_left_pixel_values.items()}
            else:
                raise ValueError(f"Unsupported `wrist_left_pixel_values` type = {type(wrist_left_pixel_values)}")
            for key, value in wrist_left_pixel_values.items():
                pixel_values[f"wrist_left_{key}"] = value

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        
        start_time=time.time()

        torch.seed()
        noise = torch.randn(1, self.future_action_window_size+1, action_dim, device=self.vlm.device)
        timestep = torch.randint(0, self.diffusion.num_timesteps, (self.future_action_window_size+1,), device=self.vlm.device)

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            outputs = super(PrismaticVLM, self.vlm).generate(
                x=noise,
                proprio=cur_robot_state,
                t=timestep,
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                gen_discret_action=False,
                ar_infer=True,
                output_scores=True, # 输出置信度
                return_dict_in_generate=True,
                **kwargs
            )
            
        logits = outputs.scores
        probs = [torch.softmax(log, dim=-1) for log in logits]
        # 获取prob列表的最后7个tensor
        last_7_tensors = probs[-7:]
        # last_10_tensors = probs[-10:]
        # 创建一个新的list，存储每个tensor中最大值的结果
        max_probs = [tensor.max().item() for tensor in last_7_tensors]
        generated_ids = outputs.sequences
        
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1) 
        actions_ar = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        mid_time = time.time()

        input_ids = input_ids[:, :-2]
        using_cfg = cfg_scale > 1.0
        # if using_cfg:
        #     noise = torch.cat([noise, noise], 0)
        #     uncondition = self.vlm.z_embedder.uncondition
        #     uncondition = uncondition.unsqueeze(0)  #[1, D]
        #     uncondition = uncondition.expand(input_ids.shape[0], 1, -1) #[B, 1, D]
        #     cfg_scale = cfg_scale
        #     sample_fn = self.vlm.forward_with_cfg
        #     model_kwargs = dict(z=uncondition, cfg_scale=cfg_scale,  input_ids=input_ids, pixel_values=pixel_values)
        # else:
        #     model_kwargs = dict(input_ids=input_ids,  pixel_values=pixel_values)
        #     sample_fn = self.vlm.forward
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.vlm.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(input_ids.shape[0], 1, -1) #[B, 1, D]
            cfg_scale = cfg_scale
            sample_fn = self.vlm.forward_with_cfg
            model_kwargs = dict(z=uncondition, cfg_scale=cfg_scale, proprio=cur_robot_state, input_ids=input_ids, pixel_values=pixel_values)
        else:
            model_kwargs = dict(input_ids=input_ids, proprio=cur_robot_state, pixel_values=pixel_values)
            sample_fn = self.vlm.forward

        print("ddim step is ", num_ddim_steps)
        # # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.ddim_diffusion is None:
                self.create_ddim(ddim_step=num_ddim_steps)
            samples = self.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                            noise.shape, 
                                                            noise, 
                                                            clip_denoised=False,
                                                            model_kwargs=model_kwargs,
                                                            progress=False,
                                                            device=self.vlm.device,
                                                            eta=0.0
                                                            )
        else:
            # DDPM Sampling
            samples = self.diffusion.p_sample_loop(sample_fn, 
                                                    noise.shape, 
                                                    noise, 
                                                    clip_denoised=False,
                                                    model_kwargs=model_kwargs,
                                                    progress=False,
                                                    device=self.vlm.device
                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()
        print("finish ddim")
        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions_diff = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        end_time = time.time()
        
        return actions_diff, actions_ar, max_probs, [mid_time - start_time, end_time - mid_time]
    
    @torch.inference_mode()
    def predict_action_batch(
        self, image: List[Image], 
        instruction: List[str], 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference in batch; maps input image and task instruction to continuous action.
        This function is used for batch inference in the simulators.
        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
        
        input_ids = []
        pixel_values = []

        # Build VLA Prompt
        B = len(image)

        if isinstance(tokenizer, LlamaTokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        for id in range(B):
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction[id].lower()}?")
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            single_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device).squeeze(0)
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            single_input_ids = torch.cat(
                (single_input_ids, torch.Tensor([29871, 2]).long().to(self.vlm.device)), dim=0
            ) # [seq]

            input_ids.append(single_input_ids)
            # Preprocess Image
            pixel_values.append(image_transform(image[id]))

        # Padding
        padding_side = "right"
        # For now, we only support Tokenizers with `padding_side = "right"`
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert padding_side == "right", f"Invalid Tokenizer `{padding_side = }`"

        model_max_length = tokenizer.model_max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, : model_max_length]
        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(pad_token_id)

        # Preprocess Image
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.vlm.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(self.vlm.device) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                attention_mask = attention_mask,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = output.hidden_states[0][-1]
        last_hidden = last_hidden[:, num_patch :]

        cumulative_sum = attention_mask.cumsum(dim=1)  
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1) #[B, D]

        assert (cognition_features.shape[0], cognition_features.shape[1]) == (B, 4096), "Batch size must be B for action prediction"
        using_cfg = cfg_scale > 1.0


        model_dtype = next(self.action_model.net.parameters()).dtype

        B = cognition_features.shape[0]
        
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,#False, try to set True 
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0)
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,#False, try to set True 
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device)
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples.cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions
    
    def create_ddim(self, ddim_step=10, noise_schedule = 'squaredcos_cap_v2', diffusion_steps = 100):
        self.ddim_diffusion = create_diffusion(timestep_respacing = "ddim"+str(ddim_step), 
                                               noise_schedule = noise_schedule,
                                               diffusion_steps = diffusion_steps, 
                                               sigma_small = True, 
                                               learn_sigma = False
                                               )
        return self.ddim_diffusion

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_proprio_stats(self, unnorm_key=None):
        """Dimensionality of the policy's proprio space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["proprio"]
    
    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

