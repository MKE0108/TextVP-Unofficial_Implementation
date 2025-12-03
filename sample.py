#!/usr/bin/env python
"""
Sample script for TextVP inference.

Usage:
    python sample.py \
        --checkpoint path/to/epoch_X.pt \
        --config path/to/train_config.json \
        --image_dir path/to/input/images \
        --output_dir path/to/output/images \
        [--device cuda:0]

Example:
    python sample.py \
        --checkpoint experiments/20251203_014400/cross_replace=0.2.../epoch_39.pt \
        --config experiments/20251203_014400/cross_replace=0.2.../train_config.json \
        --image_dir dataset/test_1130(single_data) \
        --output_dir outputs/
"""

import argparse
import os
import glob
from typing import Optional, Union, Tuple, List, Dict
import abc

import torch
import torch.nn.functional as nnf
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm import tqdm

import ptp_utils
import seq_aligner
from experiment_config import TrainConfig
from inversion import invert


# ==============================================================================
# Attention Controllers (copied from main notebook)
# ==============================================================================

MAX_NUM_WORDS = 77


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0
    
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn_uncond = attn[:h // 2]
                attn_cond = self.forward(attn[h // 2:].clone(), is_cross, place_in_unet)
                attn = torch.cat([attn_uncond, attn_cond], dim=0)
               
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, low_resource: bool = False):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.low_resource = low_resource


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, low_resource: bool = False):
        super(AttentionStore, self).__init__(low_resource)
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step].to(attn.dtype)
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn = torch.cat([attn_base.unsqueeze(0), attn_repalce_new], dim=0)
            else:
                attn = torch.cat([attn_base.unsqueeze(0), self.replace_self_attention(attn_base, attn_repalce)], dim=0)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend=None,
                 tokenizer=None,
                 device=None,
                 low_resource: bool = False):
        super(AttentionControlEdit, self).__init__(low_resource)
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        alphas = self.alphas.to(attn_base.dtype)
        attn_replace = attn_base_replace * alphas + att_replace * (1 - alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend=None, tokenizer=None, device=None, low_resource: bool = False):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, 
            local_blend, tokenizer, device, low_resource
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


# ==============================================================================
# Main Sampling Functions
# ==============================================================================

def get_torch_dtype_from_string(dtype_str):
    """Convert a string to a torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

def load_model(device: str = "cuda:0", dtype=torch.bfloat16):
    """Load Stable Diffusion model."""
    if isinstance(dtype, str):
        dtype = get_torch_dtype_from_string(dtype)
    print(f"Loading Stable Diffusion model on {device}...")
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", 
        torch_dtype=dtype
    ).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    print("Model loaded.")
    return ldm_stable


def process_single_image(
    ldm_stable,
    image_path: str,
    learned_emb: torch.Tensor,
    train_cfg: TrainConfig,
    device: str,
) -> np.ndarray:
    """Process a single image and return the edited result."""
    
    # Load and preprocess image
    before_image = Image.open(image_path).convert("RGB").resize((512, 512))
    before_latent = ptp_utils.image2latent(
        ldm_stable.vae, 
        np.array(before_image).reshape(1, 512, 512, 3)
    )
    
    # DDIM Inversion
    noised_before_latent = invert(
        ldm_stable,
        before_latent,
        prompt="",
        guidance_scale=1,
        num_inference_steps=train_cfg.num_diffusion_steps,
        device=device,
    )
    
    # Create attention controller
    controller = AttentionRefine(
        prompts=["", train_cfg.coarse_description],
        num_steps=train_cfg.num_diffusion_steps,
        cross_replace_steps=train_cfg.cross_replace_step,
        self_replace_steps=0.0,
        tokenizer=ldm_stable.tokenizer,
        device=device,
        low_resource=train_cfg.low_resource,
    )
    
    # Generate edited image
    images, _ = ptp_utils.text2image_ldm_stable_with_learned_embedding(
        ldm_stable,
        learned_emb=learned_emb,
        controller=controller,
        latent=noised_before_latent,
        num_inference_steps=train_cfg.num_diffusion_steps,
        guidance_scale=train_cfg.guidance_scale,
        low_resource=train_cfg.low_resource,
    )
    
    # Return the edited image (index 1 is the edited version)
    return images[1]


def main():
    parser = argparse.ArgumentParser(description="TextVP Sampling Script")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str, required=True,
        help="Path to the trained .pt checkpoint file"
    )
    parser.add_argument(
        "--config", "-cfg",
        type=str, required=True,
        help="Path to train_config.json"
    )
    parser.add_argument(
        "--image_dir", "-i",
        type=str, required=True,
        help="Path to directory containing input images (supports glob pattern)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str, required=True,
        help="Path to output directory for generated images"
    )
    parser.add_argument(
        "--device", "-d",
        type=str, default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--ext", "-e",
        type=str, default="png",
        help="Image extension to search for (default: png)"
    )
    # Add dtype argument for model loading
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"],
                        help="Data type for model loading")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    print(f"Loading config from: {args.config}")
    train_cfg = TrainConfig.load(args.config)
    print(f"  - coarse_description: {train_cfg.coarse_description}")
    print(f"  - cross_replace_step: {train_cfg.cross_replace_step}")
    print(f"  - guidance_scale: {train_cfg.guidance_scale}")
    
    # Load model
    device = args.device
    ldm_stable = load_model(device=device, dtype=args.dtype)
    
    # Load learned embedding
    print(f"Loading checkpoint from: {args.checkpoint}")
    learned_emb = torch.load(args.checkpoint, map_location=device)
    if hasattr(learned_emb, 'to'):
        learned_emb = learned_emb.to(device)
    print(f"  - Embedding shape: {learned_emb.shape}")
    
    # Find input images
    if os.path.isdir(args.image_dir):
        image_pattern = os.path.join(args.image_dir, f"*.{args.ext}")
    else:
        image_pattern = args.image_dir
    
    image_paths = sorted(glob.glob(image_pattern))
    if not image_paths:
        raise ValueError(f"No images found matching: {image_pattern}")
    
    print(f"\nFound {len(image_paths)} images to process")
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Generate edited image
            edited_image = process_single_image(
                ldm_stable=ldm_stable,
                image_path=image_path,
                learned_emb=learned_emb,
                train_cfg=train_cfg,
                device=device,
            )
            
            # Save result
            basename = os.path.basename(image_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(args.output_dir, f"{name}_edited{ext}")
            
            Image.fromarray(edited_image).save(output_path)
            
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue
    
    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
