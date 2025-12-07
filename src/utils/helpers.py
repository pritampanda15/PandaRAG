"""
Helper utilities for the ADMET prediction system.

Common functions for configuration loading, reproducibility,
and environment setup.
"""

import os
import random
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda", gpu_id: int = 0) -> torch.device:
    """
    Get the appropriate compute device.
    
    Args:
        preferred: Preferred device ('cuda' or 'cpu').
        gpu_id: GPU index if using CUDA.
        
    Returns:
        torch.device object.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path.
        
    Returns:
        The same path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format.
        precision: Decimal places.
        
    Returns:
        Formatted string.
    """
    for suffix in ["", "K", "M", "B"]:
        if abs(num) < 1000:
            return f"{num:.{precision}f}{suffix}"
        num /= 1000
    return f"{num:.{precision}f}T"
