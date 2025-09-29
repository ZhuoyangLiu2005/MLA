"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        images = [instance["images"] for instance in instances]
        if isinstance(images[0], dict):
            image_keys = images[0].keys()
            images = {
                k: torch.stack([img_dict[k] for img_dict in images]) 
                for k in image_keys
            }
        elif isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        else:
            raise ValueError(f"Unsupported `images` type = {type(images[0])}")
        
        next_images = [instance["next_images"] for instance in instances]
        if isinstance(next_images[0], dict):
            next_image_keys = next_images[0].keys()
            next_images = {
                k: torch.stack([img_dict[k] for img_dict in next_images]) 
                for k in next_image_keys
            }
        elif isinstance(next_images[0], torch.Tensor):
            next_images = torch.stack(next_images)
        else:
            raise ValueError(f"Unsupported `next_images` type = {type(next_images[0])}")
            
        point_cloud = [instance["point_cloud"] for instance in instances]
        if isinstance(point_cloud[0], torch.Tensor):
            point_cloud = torch.stack(point_cloud) # (B,N,3)
        next_point_cloud = [instance["next_point_cloud"] for instance in instances]
        if isinstance(next_point_cloud[0], torch.Tensor):
            next_point_cloud = torch.stack(next_point_cloud)
        
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # Adding continuous actions and batch processing.
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions)
        
        action_masks = [instance["action_masks"] for instance in instances]
        if action_masks[0] is not None:
            action_masks = torch.stack(action_masks)
        else:
            action_masks = None

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio)
        
        if "tactile" in instances[0] and instances[0]["tactile"] is not None:
            tactile = [instance["tactile"] for instance in instances]
            tactile = torch.stack(tactile)
            gripper_xyz = [instance["gripper_xyz"] for instance in instances]
            gripper_xyz = torch.stack(gripper_xyz)
        else:
            tactile = None
            gripper_xyz = None
        
        if "next_tactile" in instances[0] and instances[0]["next_tactile"] is not None:
            next_tactile = [instance["next_tactile"] for instance in instances]
            next_tactile = torch.stack(next_tactile)
        else:
            next_tactile = None

        output = dict(
            images=images,
            next_images=next_images,
            point_cloud=point_cloud,
            next_point_cloud=next_point_cloud,
            tactile=tactile,
            next_tactile=next_tactile,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_masks=action_masks,
            proprio=proprio,
            gripper_xyz=gripper_xyz,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output