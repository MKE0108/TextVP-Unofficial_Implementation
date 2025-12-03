#!/usr/bin/env python
"""
TextVP Training and Testing Script.

Usage:
    # Training
    python run_experiment.py train \
        --exp_dir new \
        --source_image path/to/before.png \
        --target_image path/to/after.png \
        --test_image_pattern "dataset/test/*.png" \
        --coarse_description "a watercolor painting"

    # Testing
    python run_experiment.py test \
        --exp_dir 20251203_014400

Example:
    python run_experiment.py train \
        --exp_dir new \
        --source_image dataset/test_1201/B_05.png \
        --target_image dataset/test_1201/A_05.png \
        --test_image_pattern "dataset/test_1130/*.png" \
        --coarse_description "a watercolor painting" \
        --guidance_scale 1 3.5 7.5 \
        --cross_replace_step 0.2 1.0 \
        --num_epochs 40
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
from experiment_config import TrainConfig, ExperimentConfig, get_or_create_exp_dir
from inversion import invert
from image_utils import ImageGrid


# ==============================================================================
# Global Variables
# ==============================================================================

MAX_NUM_WORDS = 77
device = None
ldm_stable = None
tokenizer = None


# ==============================================================================
# Attention Controllers
# ==============================================================================

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
                 local_blend=None, low_resource: bool = False):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, 
            local_blend, low_resource
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(device_name: str = "cuda:0", dtype=torch.bfloat16):
    """Load Stable Diffusion model."""
    global device, ldm_stable, tokenizer
    if isinstance(dtype, str):
        if dtype == "float16":
            dtype = torch.float16
        elif dtype == "bfloat16":
            dtype = torch.bfloat16
        elif dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype string: {dtype}")
    
    device = torch.device(device_name) if torch.cuda.is_available() else torch.device('cpu')
    print(f"Loading Stable Diffusion model on {device}...")
    
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", 
        torch_dtype=dtype
    ).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    tokenizer = ldm_stable.tokenizer
    
    print("Model loaded.")
    return ldm_stable


# ==============================================================================
# Training
# ==============================================================================

def run_training(cfg: ExperimentConfig):
    """Run training for all experiment configurations."""
    global ldm_stable, device
    
    # Generate all TrainConfig objects
    train_configs = cfg.generate_train_configs()
    
    # Filter out already completed experiments
    filtered_configs = []
    for train_cfg in train_configs:
        save_image_dir = train_cfg.get_save_dir()
        if os.path.exists(os.path.join(save_image_dir, "final.pt")):
            print(f"Skipping {save_image_dir} since final.pt exists")
            continue
        filtered_configs.append(train_cfg)
    train_configs = filtered_configs
    
    print(f"Running {len(train_configs)} experiments")
    if not train_configs:
        print("No experiments to run.")
        return
    
    exp_dir = train_configs[0].exp_dir
    print(f"Experiment directory: {exp_dir}")
    
    # Save experiment config
    cfg_save_path = os.path.join(exp_dir, "config.json")
    cfg.save(cfg_save_path)
    print(f"Config saved to {cfg_save_path}")
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    source_image = Image.open(cfg.source_image_path).convert("RGB").resize((512, 512))
    target_image = Image.open(cfg.target_image_path).convert("RGB").resize((512, 512))
    
    before_latent = ptp_utils.image2latent(ldm_stable.vae, np.array(source_image).reshape(1, 512, 512, 3))
    after_latent = ptp_utils.image2latent(ldm_stable.vae, np.array(target_image).reshape(1, 512, 512, 3))
    
    # DDIM Inversion
    print("Running DDIM inversion...")
    noised_before_latent = invert(
        ldm_stable,
        before_latent,
        prompt="",
        guidance_scale=1,
        num_inference_steps=cfg.num_diffusion_steps,
        device=device,
    )
    print("Images loaded and inverted")
    
    # Train each configuration
    for train_cfg in train_configs:
        print(f"\n{'='*60}")
        print(f"Training: {train_cfg.exp_name}")
        print(f"{'='*60}")
        
        save_image_dir = train_cfg.get_save_dir()
        os.makedirs(save_image_dir, exist_ok=True)
        
        # Save TrainConfig
        train_cfg_path = os.path.join(save_image_dir, "train_config.json")
        train_cfg.save(train_cfg_path)
        print(f"TrainConfig saved to {train_cfg_path}")
        
        # Create attention controller
        controller = AttentionRefine(
            prompts=["", train_cfg.coarse_description],
            num_steps=train_cfg.num_diffusion_steps,
            cross_replace_steps=train_cfg.cross_replace_step,
            self_replace_steps=train_cfg.self_replace_step,
        )
        
        # Setup for training
        torch.autograd.set_detect_anomaly(True)
        ldm_stable.vae.requires_grad_(False)
        ldm_stable.unet.requires_grad_(False)
        ldm_stable.text_encoder.requires_grad_(False)
        ldm_stable.scheduler = DDIMScheduler.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="scheduler")
        ldm_stable.scheduler.set_timesteps(train_cfg.num_diffusion_steps)
        
        # Select training function
        train_fn = (ptp_utils.train_text_embedding_ldm_stable 
                   if train_cfg.encoded_emb 
                   else ptp_utils.train_text_embedding_ldm_stable_with_out_encode)
        
        # Train
        learned_emb = train_fn(
            ldm_stable,
            train_cfg.coarse_description,
            controller,
            noised_before_latent,
            after_latent,
            num_steps=train_cfg.num_diffusion_steps,
            epoch=train_cfg.num_epochs,
            guidance_scale=train_cfg.guidance_scale,
            lr=train_cfg.lr,
            optimizer_cls=train_cfg.optimizer_cls,
            save_interval=train_cfg.save_interval,
            save_image_dir=save_image_dir,
            beta_weighting=train_cfg.beta_weighting
        )
        
        print(f"Training completed for: {train_cfg.exp_name}")
    
    print(f"\n{'='*60}")
    print("All training completed!")
    print(f"{'='*60}")


# ==============================================================================
# Testing
# ==============================================================================

def run_testing(exp_dir: str):
    """Run testing for all experiment configurations in the given directory."""
    global ldm_stable, device
    
    # Load experiment config
    target_exp_dir = get_or_create_exp_dir(base_dir="experiments/", exp_dir=exp_dir)
    print(f"Testing experiments in: {target_exp_dir}")
    
    cfg = ExperimentConfig.load(os.path.join(target_exp_dir, "config.json"))
    test_configs = cfg.generate_train_configs()
    all_test_image = sorted(glob.glob(cfg.test_image_pattern))
    
    print(f"Test images: {len(all_test_image)} files")
    
    if not all_test_image:
        print("No test images found!")
        return
    
    # Define column labels
    col_labels = ["Source", "Epoch0"] + [f"Epoch{e}" for e in cfg.test_epochs[1:]]
    
    # Run inference on test images
    for test_cfg in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {test_cfg.exp_name}")
        print(f"{'='*60}")
        
        exp_dir_path = test_cfg.exp_dir
        
        # Create ImageGrid
        row_labels = [os.path.basename(p).replace('.png', '').replace('.jpg', '') for p in all_test_image]
        image_grid = ImageGrid(row_labels=row_labels, col_labels=col_labels)
        
        for image_path in tqdm(all_test_image, desc="Processing images"):
            before_image = Image.open(image_path).convert("RGB").resize((512, 512))
            before_latent = ptp_utils.image2latent(ldm_stable.vae, np.array(before_image).reshape(1, 512, 512, 3))
            
            noised_before_latent = invert(
                ldm_stable,
                before_latent,
                prompt="",
                guidance_scale=1,
                num_inference_steps=test_cfg.num_diffusion_steps,
                device=device,
            )
            
            for e in test_cfg.test_epochs:
                learned_emb_path = os.path.join(exp_dir_path, test_cfg.exp_name, f"epoch_{e}.pt")
                if not os.path.exists(learned_emb_path):
                    print(f"Warning: {learned_emb_path} not found, skipping...")
                    continue
                    
                learned_emb = torch.load(learned_emb_path).to(device)
                
                controller = AttentionRefine(
                    prompts=["", test_cfg.coarse_description],
                    num_steps=test_cfg.num_diffusion_steps,
                    cross_replace_steps=test_cfg.cross_replace_step,
                    self_replace_steps=0.0,
                )
                
                images, _ = ptp_utils.text2image_ldm_stable_with_learned_embedding(
                    ldm_stable,
                    learned_emb=learned_emb,
                    controller=controller,
                    latent=noised_before_latent,
                    num_inference_steps=test_cfg.num_diffusion_steps,
                    guidance_scale=test_cfg.guidance_scale,
                    low_resource=test_cfg.low_resource
                )
                
                if e == test_cfg.test_epochs[0]:
                    image_grid.add_image([images[0], images[1]])
                else:
                    image_grid.add_image(images[1])
        
        # Save results
        save_path = os.path.join(exp_dir_path, test_cfg.exp_name, "test_image1.png")
        image_grid.save(save_path, num_rows=len(all_test_image))
        print(f"Saved images to {save_path}")
    
    # Copy results to comparison folder
    comparison_dir = os.path.join(target_exp_dir, "all_option_combinations")
    os.makedirs(comparison_dir, exist_ok=True)
    
    for result_cfg in test_configs:
        test_image_path = os.path.join(target_exp_dir, result_cfg.exp_name, "test_image1.png")
        safe_name = result_cfg.exp_name.replace('(', '').replace(')', '').replace(',', '_').replace('=', '_')
        dest_path = os.path.join(comparison_dir, f"{safe_name}_test_image1.png")
        
        if os.path.exists(test_image_path):
            os.system(f"cp '{test_image_path}' '{dest_path}'")
    
    print(f"\nResults copied to {comparison_dir}/")


# ==============================================================================
# Main
# ==============================================================================

import ast

def parse_list(value: str, item_type=float):
    """Parse string to list using ast.literal_eval."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [item_type(x) for x in parsed]
        else:
            return [item_type(parsed)]
    except (ValueError, SyntaxError):
        # Try splitting by comma
        return [item_type(x.strip()) for x in value.split(',')]


def parse_cross_replace_step(value: str):
    """Parse cross_replace_step from string like '[[0.2,1.0]]' or '0.2,1.0'."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                return parsed  # Already [[0.2, 1.0], ...]
            else:
                return [parsed]  # [0.2, 1.0] -> [[0.2, 1.0]]
        else:
            return [[0.2, 1.0]]
    except (ValueError, SyntaxError):
        # Try parsing "0.2,1.0" format
        parts = [float(x.strip()) for x in value.split(',')]
        if len(parts) == 2:
            return [[parts[0], parts[1]]]
        return [[0.2, 1.0]]


def parse_bool_list(value: str):
    """Parse string to bool list."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [bool(x) for x in parsed]
        else:
            return [bool(parsed)]
    except (ValueError, SyntaxError):
        return [value.lower() in ('true', '1', 'yes')]


def main():
    parser = argparse.ArgumentParser(
        description="TextVP Training and Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python run_experiment.py --mode train \\
      --source_image dataset/B_05.png \\
      --target_image dataset/A_05.png \\
      --test_image_pattern "dataset/test/*.png" \\
      --coarse_description "a watercolor painting" \\
      --guidance_scale "[1, 3.5, 7.5]" \\
      --cross_replace_step "[[0.2, 1.0]]" \\
      --num_epochs 40

  # Testing
  python run_experiment.py --mode test --exp_dir 20251203_014400
  
  # Full (Train + Test)
  python run_experiment.py --mode full \\
      --source_image dataset/B_05.png \\
      --target_image dataset/A_05.png \\
      --test_image_pattern "dataset/test/*.png" \\
      --coarse_description "a watercolor painting" \\
      --guidance_scale "[1, 3.5, 7.5]" \\
      --num_epochs 40
        """
    )
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "test", "full"], required=True,
                        help="Mode: 'train', 'test', or 'full' (train + test)")
    
    # Directory settings
    parser.add_argument("--exp_dir", type=str, default="new",
                        help="Experiment directory name ('new' to create with timestamp, or specific name for test)")
    parser.add_argument("--base_dir", type=str, default="experiments/",
                        help="Base directory for experiments")
    
    # Image paths
    parser.add_argument("--source_image", type=str, default="",
                        help="Path to source (BEFORE) image")
    parser.add_argument("--target_image", type=str, default="",
                        help="Path to target (AFTER) image")
    parser.add_argument("--test_image_pattern", type=str, default="",
                        help="Glob pattern for test images")
    
    # Model settings
    parser.add_argument("--low_resource", type=str, default="False",
                        help="Enable low resource mode (for 12GB GPU)")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    
    # Experiment parameters (search space) - all as strings
    parser.add_argument("--guidance_scale", type=str, default="[7.5]",
                        help="Guidance scale values, e.g. '[1, 3.5, 7.5]'")
    parser.add_argument("--cross_replace_step", type=str, default="[[0.2, 1.0]]",
                        help="Cross replace step ranges, e.g. '[[0.2, 1.0]]'")
    parser.add_argument("--self_replace_step", type=str, default="[0.0]",
                        help="Self replace step values, e.g. '[0.0]'")
    parser.add_argument("--encoded_emb", type=str, default="[False]",
                        help="Encoded embedding options, e.g. '[False]' or '[True, False]'")
    
    # Style description
    parser.add_argument("--coarse_description", type=str, default="a watercolor painting",
                        help="Coarse style description for initialization")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW", "SGD"],
                        help="Optimizer to use")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--beta_weighting", type=str, default="True",
                        help="Use beta weighting for loss")
    parser.add_argument("--test_epochs", type=str, default="[0, 5, 10, 20, 30, 40, 49]",
                        help="Epochs to test, e.g. '[0, 5, 10, 20, 30, 40, 49]'")
    # Add dtype argument for model loading
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"],
                        help="Data type for model loading")
    args = parser.parse_args()
    
    # Load model
    load_model(args.device, dtype=args.dtype)
    
    if args.mode == "train":
        # Validate required arguments
        if not args.source_image or not args.target_image:
            parser.error("--source_image and --target_image are required for training")
        
        # Map optimizer string to class
        optimizer_map = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        optimizer_cls = optimizer_map.get(args.optimizer, torch.optim.AdamW)
        
        # Parse string arguments to lists
        guidance_scale_list = parse_list(args.guidance_scale, float)
        cross_replace_steps = parse_cross_replace_step(args.cross_replace_step)
        self_replace_step_list = parse_list(args.self_replace_step, float)
        encoded_emb_list = parse_bool_list(args.encoded_emb)
        test_epochs_list = parse_list(args.test_epochs, int)
        low_resource = False
        beta_weighting = args.beta_weighting.lower() in ('true', '1', 'yes')
        
        print(f"Parsed parameters:")
        print(f"  guidance_scale: {guidance_scale_list}")
        print(f"  cross_replace_step: {cross_replace_steps}")
        print(f"  self_replace_step: {self_replace_step_list}")
        print(f"  encoded_emb: {encoded_emb_list}")
        print(f"  test_epochs: {test_epochs_list}")
        
        # Create ExperimentConfig
        cfg = ExperimentConfig(
            exp_dir=args.exp_dir,
            base_dir=args.base_dir,
            source_image_path=args.source_image,
            target_image_path=args.target_image,
            test_image_pattern=args.test_image_pattern,
            low_resource=low_resource,
            num_diffusion_steps=args.num_diffusion_steps,
            option_guidance_scale=guidance_scale_list,
            option_cross_replace_step=cross_replace_steps,
            option_self_replace_step=self_replace_step_list,
            option_encoded_emb=encoded_emb_list,
            coarse_description=args.coarse_description,
            lr=args.lr,
            optimizer_cls=optimizer_cls,
            num_epochs=args.num_epochs,
            save_interval=args.save_interval,
            beta_weighting=beta_weighting,
            test_epochs=test_epochs_list,
        )
        
        run_training(cfg)
        
    elif args.mode == "test":
        exp_dir = args.exp_dir if args.exp_dir != "new" else None
        run_testing(exp_dir)
    
    elif args.mode == "full":
        # Validate required arguments
        if not args.source_image or not args.target_image:
            parser.error("--source_image and --target_image are required for full mode")
        
        # Map optimizer string to class
        optimizer_map = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        optimizer_cls = optimizer_map.get(args.optimizer, torch.optim.AdamW)
        
        # Parse string arguments to lists
        guidance_scale_list = parse_list(args.guidance_scale, float)
        cross_replace_steps = parse_cross_replace_step(args.cross_replace_step)
        self_replace_step_list = parse_list(args.self_replace_step, float)
        encoded_emb_list = parse_bool_list(args.encoded_emb)
        test_epochs_list = parse_list(args.test_epochs, int)
        low_resource = False
        beta_weighting = args.beta_weighting.lower() in ('true', '1', 'yes')
        
        print(f"Parsed parameters:")
        print(f"  guidance_scale: {guidance_scale_list}")
        print(f"  cross_replace_step: {cross_replace_steps}")
        print(f"  self_replace_step: {self_replace_step_list}")
        print(f"  encoded_emb: {encoded_emb_list}")
        print(f"  test_epochs: {test_epochs_list}")
        
        # Create ExperimentConfig
        cfg = ExperimentConfig(
            exp_dir=args.exp_dir,
            base_dir=args.base_dir,
            source_image_path=args.source_image,
            target_image_path=args.target_image,
            test_image_pattern=args.test_image_pattern,
            low_resource=low_resource,
            num_diffusion_steps=args.num_diffusion_steps,
            option_guidance_scale=guidance_scale_list,
            option_cross_replace_step=cross_replace_steps,
            option_self_replace_step=self_replace_step_list,
            option_encoded_emb=encoded_emb_list,
            coarse_description=args.coarse_description,
            lr=args.lr,
            optimizer_cls=optimizer_cls,
            num_epochs=args.num_epochs,
            save_interval=args.save_interval,
            beta_weighting=beta_weighting,
            test_epochs=test_epochs_list,
        )
        
        # Run training
        print("\n" + "="*60)
        print("PHASE 1: TRAINING")
        print("="*60)
        run_training(cfg)
        
        # Run testing on the same experiment
        print("\n" + "="*60)
        print("PHASE 2: TESTING")
        print("="*60)

        run_testing(cfg.exp_dir)
        
        print("\n" + "="*60)
        print("FULL PIPELINE COMPLETED!")
        print("="*60)


if __name__ == "__main__":
    main()
