# src/pipelines/utils.py

from omegaconf import DictConfig
from typing import Tuple
import numpy as np
import torch
from src.training.utils import get_device
from src.models.cnn import KeywordSpottingNet


def load_cmvn(config: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CMVN (mean and std) statistics for pipeline evaluation or realtime.

    Args:
        config (DictConfig): Hydra configuration object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Loaded (mean, std) arrays.

    Raises:
        FileNotFoundError: If CMVN files cannot be found at the configured paths.
        RuntimeError: If CMVN files exist but cannot be loaded or are invalid.
    """
    mean_path = config.get("realtime", {}).get("default_mean_path", None)
    std_path = config.get("realtime", {}).get("default_std_path", None)

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

    print(f"[CMVN] Loaded mean/std from {mean_path} and {std_path}")
    return mean, std


def load_model(config: DictConfig, num_classes: int) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a model compatible with the current device (CPU / GPU).
    
    A model is instantiated and its weights are loaded from a device-appropriate 
    checkpoint (CPU or GPU). The model is moved to the selected device and set to evaluation mode.

    Args:
        config (DictConfig): Hydra configuration containing runtime and
            checkpoint path settings.
        num_classes (int): Number of output classes the model was trained on.
            Must match the checkpoint.

    Returns:
        Tuple[torch.nn.Module, torch.device]:
            - The model ready for inference.
            - The torch device the model is loaded onto.

    Raises:
        FileNotFoundError: If no model checkpoint is provided at the expected path.
    """
    device = get_device(config)

    model = KeywordSpottingNet(config=config, num_classes=num_classes)

    if device.type == "cpu":
        model_path = config.get("realtime", {}).get("default_model_path_cpu", None)
    else:
        model_path = config.get("realtime", {}).get("default_model_path_gpu", None)

    if model_path is None:
        raise FileNotFoundError(
            "No model checkpoint path provided for realtime inference.\n"
            f"Expected a '{device.type}' compatible checkpoint.\n"
            f"Make sure 'best_model_{device.type}.pt' exists at the configured path."
        )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"[Model] Loaded model from {model_path} onto {device}")
    return model, device