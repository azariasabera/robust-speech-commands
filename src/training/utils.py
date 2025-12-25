# src/training/utils.py

from omegaconf import DictConfig
from typing import Any, Optional
import torch

def get_training_param(
    config: DictConfig, 
    param: Optional[str] = None, 
    key: str = "", 
    default: Any = None
) -> Any:
    """
    Retrieve a training parameter from a Hydra DictConfig with a safe fallback.

    Args:
        config: Hydra DictConfig object.
        param: Optional subsection under 'training'. If None or empty, key is accessed directly under 'training'.
        key: Parameter name to retrieve.
        default: Value to return if key is missing.

    Returns:
        The value from config.training[key] or config.training[param][key] if param is provided,
        otherwise default.
    """
    training = config.get("training", {})

    if param:
        return training.get(param, {}).get(key, default)

    return training.get(key, default)

def get_device(config: DictConfig) -> torch.device:
    """
    Determine the device to run the model on based on the config.

    Args:
        config: Hydra DictConfig object.

    Returns:
        torch.device: 'cuda' if available and auto-selected, else 'cpu'.
    """
    device = get_training_param(config=config, key="device", default="auto")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)