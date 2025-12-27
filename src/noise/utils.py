# src/noise/utils.py

from omegaconf import DictConfig
from typing import Optional, Any

def get_noise_param(config: DictConfig, param: Optional[str] = None, 
                    key: str = "", default: Any = None) -> Any:
    """
    Retrieve a parameter from config.noise with a safe fallback.

    Args:
        config: Hydra DictConfig object.
        param: Optional subsection under 'noise'.
        key: Parameter name to retrieve.
        default: Value to return if key is missing.

    Returns:
        The value from config.noise[key] or config.noise[param][key] or default.
    """
    noise_cfg = config.get("noise", {})

    if param:
        return noise_cfg.get(param, {}).get(key, default)

    return noise_cfg.get(key, default)