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
Experiment configuration utilities.

This module provides two configuration classes:
- ExperimentConfig: Main configuration with experiment options (search space lists)
- TrainConfig: Individual training configuration with single values from the search space

Usage:
    # Create main experiment config with search space
    exp_cfg = ExperimentConfig(
        option_tau=[0.1, 0.2, 0.3],
        option_encoded_emb=[False],
        ...
    )
    
    # Generate individual training configs
    train_configs = exp_cfg.generate_train_configs()
    
    # Each train_config has single values
    for train_cfg in train_configs:
        print(train_cfg.tau, train_cfg.encoded_emb)
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import os
import glob
import torch
from image_utils import view_images, save_images


# ==============================================================================
# TrainConfig: Individual training configuration with single values
# ==============================================================================

@dataclass
class TrainConfig:
    """
    Configuration for a single training run.
    Contains specific parameter values (not lists) for one experiment.
    """
    # ============== Directory Configuration ==============
    exp_dir: str = ""  # Resolved experiment directory path
    exp_name: str = ""  # Individual experiment name
    
    # ============== Image Paths ==============
    source_image_path: str = ""
    target_image_path: str = ""
    test_image_pattern: str = ""
    
    # ============== Model Configuration ==============
    low_resource: bool = False
    num_diffusion_steps: int = 50
    
    
    # ============== Current Experiment Parameters (single values) ==============
    cross_replace_step: float = 0.2  # cross_replace_steps value
    guidance_scale: float = 7.5
    encoded_emb: bool = False
    self_replace_step: float = 0.4  # self_replace_steps value
    
    # ============== Style Description ==============
    coarse_description: str = "A watercolor painting"
    
    # ============== Training Parameters ==============
    lr: float = 1e-3
    optimizer_cls: Any = None
    num_epochs: int = 40
    save_interval: int = 1
    beta_weighting: bool = True
    
    # ============== Test Parameters ==============
    test_epochs: List[int] = field(default_factory=lambda: [0, 5, 10, 20, 30, 39])
    
    # ============== Internal State ==============
    image_record: List = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        if self.optimizer_cls is None:
            self.optimizer_cls = torch.optim.Adam
    
    @property
    def cross_replace_steps(self) -> float:
        """Alias for tau (backward compatibility)."""
        return self.cross_replace_step
    
    def get_save_dir(self) -> str:
        """Get the full save directory path for this experiment."""
        return os.path.join(self.exp_dir, self.exp_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k != 'image_record'}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Create config from dictionary."""
        filtered = {k: v for k, v in d.items() 
                   if not k.startswith('_') and k != 'image_record'}
        return cls(**filtered)
    
    def copy(self, **overrides) -> "TrainConfig":
        """Create a copy of this config with optional overrides."""
        d = self.to_dict()
        d.update(overrides)
        return TrainConfig.from_dict(d)
    
    def save(self, path: str):
        """Save config to JSON file."""
        import json
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not callable(v) and k != 'optimizer_cls' 
                      and not k.startswith('_') and k != 'image_record'}
        config_dict['optimizer_cls'] = str(self.optimizer_cls)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Handle optimizer_cls
        if 'optimizer_cls' in config_dict:
            opt_str = config_dict['optimizer_cls']
            if 'Adam' in opt_str and 'AdamW' not in opt_str:
                config_dict['optimizer_cls'] = torch.optim.Adam
            elif 'AdamW' in opt_str:
                config_dict['optimizer_cls'] = torch.optim.AdamW
            elif 'SGD' in opt_str:
                config_dict['optimizer_cls'] = torch.optim.SGD
            else:
                config_dict['optimizer_cls'] = torch.optim.Adam
        return cls(**config_dict)
    
    # ============== Image Recording Methods ==============
    def add_image(self, image):
        """Add image(s) to the record."""
        if type(image) is not list:
            image = [image]
        self.image_record.extend(image)
    
    def show_images(self, num_rows: int, row_labels: Optional[List[str]] = None, 
                    col_labels: Optional[List[str]] = None):
        """Display recorded images in a grid."""
        view_images(self.image_record, num_rows=num_rows, 
                   row_labels=row_labels, col_labels=col_labels)
    
    def save_images(self, save_path: str, num_rows: int = 1, 
                    row_labels: Optional[List[str]] = None, 
                    col_labels: Optional[List[str]] = None):
        """Save recorded images to a file."""
        save_images(self.image_record, save_path=save_path, num_rows=num_rows, 
                   row_labels=row_labels, col_labels=col_labels)
    
    def clear_images(self):
        """Clear the image record."""
        self.image_record = []
    
    def __repr__(self) -> str:
        return (f"TrainConfig(exp_name={self.exp_name}, "
                f"cross_replace_step={self.cross_replace_step}, "
                f"self_replace_step={self.self_replace_step}, "
                f"encoded_emb={self.encoded_emb}, "
                f"guidance_scale={self.guidance_scale})")


# ==============================================================================
# ExperimentConfig: Main configuration with search space options
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Main experiment configuration with search space options (lists).
    Use generate_train_configs() to create individual TrainConfig objects.
    """
    # ============== Directory Configuration ==============
    exp_dir: str = "new"  # "new" to create new, None to use latest, or specific name
    base_dir: str = "experiments"
    
    # ============== Image Paths ==============
    source_image_path: str = ""
    target_image_path: str = ""
    test_image_pattern: str = ""
    
    # ============== Model Configuration ==============
    low_resource: bool = False
    num_diffusion_steps: int = 50
    
    # ============== Experiment Options (Search Space - Lists) ==============
    option_guidance_scale: List[float] = field(default_factory=lambda: [7.5])
    option_cross_replace_step: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    option_encoded_emb: List[bool] = field(default_factory=lambda: [False])
    option_self_replace_step: List[float] = field(default_factory=lambda: [0.4])
    # ============== Style Description ==============
    coarse_description: str = "A watercolor painting"
    
    # ============== Training Parameters ==============
    lr: float = 1e-3
    optimizer_cls: Any = None
    num_epochs: int = 50
    save_interval: int = 1
    beta_weighting: bool = True
    
    # ============== Test Parameters ==============
    test_epochs: List[int] = field(default_factory=lambda: [0, 5, 10, 20, 30, 39])
    
    # ============== Experiment Name Template ==============
    exp_name_template: str = "cross_replace={cross_replace_step},self_replace={self_replace_step},encoded={encoded_emb},guidance_scale={guidance_scale},"
    
    # ============== Internal State ==============
    _resolved_exp_dir: str = field(default="", init=False, repr=False)
    
    def __post_init__(self):
        if self.optimizer_cls is None:
            self.optimizer_cls = torch.optim.Adam
        if self.exp_dir =="new":
            resolved_dir = get_or_create_exp_dir(
                base_dir=self.base_dir,
                exp_dir=self.exp_dir
            )
            self.exp_dir= resolved_dir.split(os.sep)[-1]
    
    def resolve_exp_dir(self) -> str:
        """
        Resolve and return the actual experiment directory path.
        Creates new directory if needed.
        """
        if self._resolved_exp_dir:
            return self._resolved_exp_dir
        self._resolved_exp_dir = get_or_create_exp_dir(
            base_dir=self.base_dir, 
            exp_dir=self.exp_dir
        )
        return self._resolved_exp_dir
    
    def generate_train_configs(
        self,
        exp_dir_override: Optional[str] = None,
    ) -> List[TrainConfig]:
        """
        Generate TrainConfig objects for all parameter combinations.
        
        Args:
            exp_dir_override: Override experiment directory. 
                              If None, use self.exp_dir
        
        Returns:
            List of TrainConfig objects
        """
        # Handle directory override
        if exp_dir_override is not None:
            resolved_dir = get_or_create_exp_dir(
                base_dir=self.base_dir,
                exp_dir=exp_dir_override
            )
        else:
            resolved_dir = self.resolve_exp_dir()
        
        
        configs = []
        for cross_replace_step in self.option_cross_replace_step:
            for encoded_emb in self.option_encoded_emb:
               
                for self_replace_step in self.option_self_replace_step:
                    for guidance_scale in self.option_guidance_scale:
                        exp_name = self.exp_name_template.format(
                                cross_replace_step=cross_replace_step,
                                encoded_emb=encoded_emb,
                                self_replace_step=self_replace_step,
                                guidance_scale=guidance_scale
                        )
                        
                        train_cfg = TrainConfig(
                            # Directory
                            exp_dir=resolved_dir,
                            exp_name=exp_name,
                            # Image paths
                            source_image_path=self.source_image_path,
                            target_image_path=self.target_image_path,
                            test_image_pattern=self.test_image_pattern,
                            # Model config
                            low_resource=self.low_resource,
                            num_diffusion_steps=self.num_diffusion_steps,
                            # Experiment parameters (single values)
                            guidance_scale=guidance_scale,
                            cross_replace_step=cross_replace_step,
                            encoded_emb=encoded_emb,
                            self_replace_step=self_replace_step,
                            # Style
                            coarse_description=self.coarse_description,
                            # Training params
                            lr=self.lr,
                            optimizer_cls=self.optimizer_cls,
                            num_epochs=self.num_epochs,
                            save_interval=self.save_interval,
                            beta_weighting=self.beta_weighting,
                            # Test params
                            test_epochs=self.test_epochs.copy(),
                        )
                        configs.append(train_cfg)
        
        return configs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        filtered = {k: v for k, v in d.items() if not k.startswith('_')}
        return cls(**filtered)
    
    def save(self, path: str):
        """Save config to JSON file."""
        import json
        import shutil
        
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not callable(v) and k != 'optimizer_cls' 
                      and not k.startswith('_')}
        config_dict['optimizer_cls'] = str(self.optimizer_cls)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Copy train_image and all test images to exp folder
        exp_dir = os.path.dirname(path)
        images_dir = os.path.join(exp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Copy source image
        if self.source_image_path and os.path.exists(self.source_image_path):
            dst_path = os.path.join(images_dir, 'source_' + os.path.basename(self.source_image_path))
            shutil.copy2(self.source_image_path, dst_path)
            print(f"Copied source image to: {dst_path}")
        
        # Copy target image
        if self.target_image_path and os.path.exists(self.target_image_path):
            dst_path = os.path.join(images_dir, 'target_' + os.path.basename(self.target_image_path))
            shutil.copy2(self.target_image_path, dst_path)
            print(f"Copied target image to: {dst_path}")
        
        # Copy all test images matching the pattern
        if self.test_image_pattern:
            test_images = glob.glob(self.test_image_pattern)
            if(len(test_images) == 0):
                print(f"Warning!!: No test images found matching pattern: {self.test_image_pattern}")
            for i, test_img in enumerate(test_images):
                if os.path.exists(test_img):
                    dst_path = os.path.join(images_dir, f'test_{i:03d}_' + os.path.basename(test_img))
                    shutil.copy2(test_img, dst_path)
            if test_images:
                print(f"Copied {len(test_images)} test images to: {images_dir}")

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Handle optimizer_cls
        if 'optimizer_cls' in config_dict:
            opt_str = config_dict['optimizer_cls']
            if 'Adam' in opt_str and 'AdamW' not in opt_str:
                config_dict['optimizer_cls'] = torch.optim.Adam
            elif 'AdamW' in opt_str:
                config_dict['optimizer_cls'] = torch.optim.AdamW
            elif 'SGD' in opt_str:
                config_dict['optimizer_cls'] = torch.optim.SGD
            else:
                config_dict['optimizer_cls'] = torch.optim.Adam
        return cls(**config_dict)
    
    def copy(self, **overrides) -> "ExperimentConfig":
        """Create a copy of this config with optional overrides."""
        d = self.to_dict()
        d.update(overrides)
        new_cfg = ExperimentConfig.from_dict(d)
        # Don't copy resolved dir - let it resolve fresh
        return new_cfg
    
    def __repr__(self) -> str:
        return (f"ExperimentConfig(\n"
                f"  option_guidance_scale={self.option_guidance_scale},\n"
                f"  option_self_replace_step={self.option_self_replace_step},\n"
                f"  option_encoded_emb={self.option_encoded_emb},\n"
                f"  num_epochs={self.num_epochs}, lr={self.lr}\n"
                f")")


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_or_create_exp_dir(base_dir: str = "experiments", exp_dir: Optional[str] = None) -> str:
    """
    Get or create experiment directory.
    
    Args:
        base_dir: Base directory for all experiments
        exp_dir: "new" to create new, None to find latest, or specific name
    
    Returns:
        Path to the experiment directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    if exp_dir == "new" or exp_dir is None:
        # Find existing experiment directories (format: YYYYMMDD_HHMMSS)
        existing_dirs = glob.glob(os.path.join(base_dir, "[0-9]*_[0-9]*"))
        existing_dirs = [d for d in existing_dirs if os.path.isdir(d)]
        
        if exp_dir is None and existing_dirs:
            # Return the latest experiment directory
            latest_dir = max(existing_dirs, key=os.path.getctime)
            print(f"Using existing experiment directory: {latest_dir}")
            return latest_dir
        else:
            # Create new experiment directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir = os.path.join(base_dir, timestamp)
            os.makedirs(new_dir, exist_ok=True)
            print(f"Created new experiment directory: {new_dir}")
            return new_dir
    else:
        # Use specific experiment directory
        specific_dir = os.path.join(base_dir, exp_dir)
        os.makedirs(specific_dir, exist_ok=True)
        return specific_dir


# ==============================================================================
# Backward Compatibility
# ==============================================================================

# Alias for backward compatibility
GlobalConfig = ExperimentConfig
config = TrainConfig

def generate_experiment_configs(
    cfg: Optional[ExperimentConfig] = None,
    global_config: Optional[ExperimentConfig] = None,
    exp_dir: Optional[str] = None,
    base_dir: str = "experiments",
    **kwargs
) -> List[TrainConfig]:
    """
    Generate training configs from experiment config.
    
    Backward compatibility wrapper for ExperimentConfig.generate_train_configs()
    """
    if cfg is None:
        cfg = global_config
    if cfg is None:
        cfg = ExperimentConfig()
    
    if exp_dir is not None:
        return cfg.generate_train_configs(exp_dir_override=exp_dir)
    else:
        return cfg.generate_train_configs()
