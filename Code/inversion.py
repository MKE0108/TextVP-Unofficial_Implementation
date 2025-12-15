# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DDIM Inversion and Sampling utilities for Stable Diffusion.

This module provides functions for inverting images to latent space
and sampling from latent space using DDIM scheduler.
"""

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

import ptp_utils as ptp_utils


@torch.no_grad()
def invert(
    pipe,
    start_latents: torch.Tensor,
    prompt: str = "",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 80,
    negative_prompt: str = "",
    device: str = "cuda",
):
    """
    Invert an image latent to noise using DDIM inversion.
    
    Args:
        pipe: StableDiffusionPipeline instance
        start_latents: Starting latent representation of the image
        prompt: Text prompt (usually empty for inversion)
        guidance_scale: Guidance scale for classifier-free guidance
        num_inference_steps: Number of diffusion steps
        negative_prompt: Negative prompt
        device: Device to use
    
    Returns:
        Inverted (noised) latents
    """
    ptp_utils.register_attention_control(pipe, None)
    ori_scheduler = pipe.scheduler
    invert_scheduler = DDIMInverseScheduler.from_pretrained(
        'sd-legacy/stable-diffusion-v1-5', 
        subfolder="scheduler"
    )
    pipe.scheduler = invert_scheduler
    pipe.to(device)
    
    inv_latents, _ = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        output_type='latent',
        return_dict=False,
        num_inference_steps=num_inference_steps,
        latents=start_latents,
    )
    
    pipe.scheduler = ori_scheduler
    return inv_latents


@torch.no_grad()
def sample(
    pipe,
    start_latents: torch.Tensor,
    prompt: str,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 80,
    negative_prompt: str = "",
    do_classifier_free_guidance: bool = True,
    device: str = "cuda",
):
    """
    Sample from noise latents using DDIM scheduler.
    
    Args:
        pipe: StableDiffusionPipeline instance
        start_latents: Starting noise latents
        prompt: Text prompt for generation
        guidance_scale: Guidance scale for classifier-free guidance
        num_inference_steps: Number of diffusion steps
        negative_prompt: Negative prompt
        do_classifier_free_guidance: Whether to use classifier-free guidance
        device: Device to use
    
    Returns:
        Generated latents
    """
    ptp_utils.register_attention_control(pipe, None)
    ori_scheduler = pipe.scheduler
    scheduler = DDIMScheduler.from_pretrained(
        'sd-legacy/stable-diffusion-v1-5', 
        subfolder="scheduler"
    )
    pipe.scheduler = scheduler
    pipe.to(device)
    
    samples, _ = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        output_type='latent',
        return_dict=False,
        num_inference_steps=num_inference_steps,
        latents=start_latents,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )
    
    pipe.scheduler = ori_scheduler
    return samples
