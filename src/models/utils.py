# src/models/utils.py

from omegaconf import DictConfig
from typing import Any

def get_cnn_param(config: DictConfig, key: str, default: Any) -> Any:
    """
    Get a parameter from config.cnn with a fallback default.

    Args:
        config: Hydra DictConfig object
        key: Parameter name
        default: Value to return if key is missing

    Returns:
        The value from config.cnn[key] or default
    """

    return config.get("cnn", {}).get(key, default)