# src/training/utils.py

from omegaconf import DictConfig
from typing import Any, Optional

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