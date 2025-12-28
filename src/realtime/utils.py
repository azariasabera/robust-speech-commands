# src/realtime/utils.py

from typing import Any
from omegaconf import DictConfig

def get_realtime_param(config: DictConfig, key: str, default: Any = None) -> Any:
    """
    Safely get a realtime parameter from config.

    Args:
        config (DictConfig): Hydra configuration object.
        key (str): Parameter name to retrieve from 'config.realtime'.
        default (Any): Value to return if the key does not exist.

    Returns:
        Any: The value of the parameter from 'config.realtime' or the default.
    """
    return config.get("realtime", {}).get(key, default)