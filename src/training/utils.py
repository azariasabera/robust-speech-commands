# src/training/utils.py

from omegaconf import DictConfig
from typing import Any, Optional
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np


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

def plot_history(history: dict, config: DictConfig) -> None:
    """
    Plot training and validation loss/accuracy and optionally save the plot.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        config: Hydra DictConfig object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    if get_training_param(config=config, param="plotting", key="save", default=True):
        plot_dir = Path(get_training_param(
            config=config, param="plotting", key="dir", default=Path.cwd()/"saved_plot"
        ))
        plot_dir.mkdir(parents=True, exist_ok=True)
        filename = get_training_param(config=config, param="plotting", key="filename", default="plot.png")
        
        plot_pth = plot_dir / filename
        curr = datetime.now().strftime("%Y%m%d_%H%M%S") # to avoid overwriting saved plot
        plot_pth = plot_dir / f"{plot_pth.stem}_{curr}{plot_pth.suffix}"
        plt.savefig(plot_pth)

    if get_training_param(config=config, param="plotting", key="show", default=True):
        plt.show()

def set_seed(config: DictConfig) -> None: 
    """ Set seed for reproducibility for torch, numpy, and python.random. 
    Args: 
        config: Hydra DictConfig object. 
    """ 
    seed = get_training_param(config=config, key="seed", default=42) 
    
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    
    if get_training_param(config=config, key="deterministic", default=False): 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

def load_data(config: DictConfig, X: np.ndarray, y: np.ndarray) -> DataLoader:
    """
    Convert numpy arrays to a PyTorch DataLoader.

    Args:
        config: Hydra DictConfig object.
        X: Input features, shape (N, H, W)
        y: Class labels, shape (N,)

    Returns:
        DataLoader yielding batches of (X, y) tensors suitable for CNN training.
    """
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    y_tensor = torch.from_numpy(y).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    batch_size = get_training_param(config=config, key="batch_size", default=64)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return loader