# src/realtime/utils.py

from typing import Any, Optional, Tuple
from omegaconf import DictConfig
import numpy as np

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

def load_cmvn(config: DictConfig, mean: Optional[np.ndarray] = None,
            std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CMVN (mean and std) statistics for realtime inference.

    Args:
        config (DictConfig): Hydra configuration object.
        mean (Optional[np.ndarray]): Preloaded mean array.
        std (Optional[np.ndarray]): Preloaded std array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Loaded (mean, std) arrays.

    Raises:
        FileNotFoundError: If CMVN files cannot be found at the configured paths.
        RuntimeError: If CMVN files exist but cannot be loaded or are invalid.
    """
    if mean is not None and std is not None:
        return mean, std

    mean_path = get_realtime_param(config, "default_mean_path")
    std_path = get_realtime_param(config, "default_std_path")

    try:
        mean = np.load(mean_path)["arr_0"]
        std = np.load(std_path)["arr_0"]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "CMVN file(s) not found at configured paths.\n"
            f"mean_path={mean_path}\n"
            f"std_path={std_path}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Failed to load CMVN statistics.\n"
            "Ensure the files are valid NumPy .npz files containing 'arr_0'."
        ) from e

    return mean, std