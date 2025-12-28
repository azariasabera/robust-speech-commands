# src/realtime/utils.py

from typing import Any, Optional, Tuple
from omegaconf import DictConfig
import numpy as np
import torch
from src.models.cnn import KeywordSpottingNet
from src.training.utils import get_device

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

def load_realtime_model(
        config: DictConfig, num_classes: int, 
        model: Optional[torch.nn.Module] = None
    ) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a model compatible with the current device (CPU / GPU).
    
    If a model instance is not provided, a new model is instantiated and its
    weights are loaded from a device-appropriate checkpoint (CPU or GPU).
    The model is moved to the selected device and set to evaluation mode.

    Args:
        config (DictConfig): Hydra configuration containing runtime and
            checkpoint path settings.
        num_classes (int): Number of output classes the model was trained on.
            Must match the checkpoint.
        model (Optional[torch.nn.Module]): Pre-instantiated model. If provided,
            checkpoint loading is skipped and the model is only moved to the
            selected device and set to eval mode.

    Returns:
        Tuple[torch.nn.Module, torch.device]:
            - The model ready for inference.
            - The torch device the model is loaded onto.

    Raises:
        FileNotFoundError: If no model checkpoint is provided at the expected path.
    """
    device = get_device(config)
    device_type = device.type

    if model is None:
        model = KeywordSpottingNet(config=config, num_classes=num_classes)

        # Select checkpoint based on device
        if device_type == "cpu":
            model_path = get_realtime_param(config, "default_model_path_cpu")
        else:
            model_path = get_realtime_param(config, "default_model_path_gpu")

        if model_path is None:
            raise FileNotFoundError(
                "No model checkpoint path provided for realtime inference.\n"
                f"Expected a '{device_type}' compatible checkpoint.\n"
                f"Make sure 'best_model_{device_type}.pt' exists at the configured path."
            )

        # SAFE loading across devices
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device