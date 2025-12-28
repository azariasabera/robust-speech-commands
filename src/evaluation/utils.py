# src/evaluation/utils.py

from typing import Any, Optional
from omegaconf import DictConfig
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

def get_eval_param(config: DictConfig, param: Optional[str] = None, 
                   key: str = "", default: Any = None) -> Any:
    """
    Safely retrieve an evaluation parameter from the config.

    Args:
        config (DictConfig): Hydra configuration object.
        param (Optional[str]): Subsection under 'evaluation'.
        key (str): Parameter key to retrieve.
        default (Any): Default value if key is missing.

    Returns:
        Any: Parameter value from config or default.
    """
    evaluation = config.get("evaluation", {})

    if param:
        return evaluation.get(param, {}).get(key, default)

    return evaluation.get(key, default)


def plot_confusion(config: DictConfig, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    """
    Plot confusion matrix figure if enabled in config and optionally save the matrix as picture.

    Args:
        config (DictConfig): Hydra configuration object.
        y_true (torch.Tensor): True labels (1D tensor).
        y_pred (torch.Tensor): Predicted labels (1D tensor).
    """
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if get_eval_param(config=config, param="plotting", key="save", default=True):
            
        save_dir = Path(get_eval_param(config=config, param="plotting", key="dir", default="plots"))
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = get_eval_param(config=config, param="plotting", key="filename", default="confusion_matrix.png")
        save_pth = save_dir / filename
        curr = datetime.now().strftime("%Y%m%d_%H%M%S") # to avoid overwriting saved confusion matrix
        save_pth = save_dir / f"{save_pth.stem}_{curr}{save_pth.suffix}"

        plt.savefig(save_pth)
    
    if get_eval_param(config=config, param="plotting", key="show", default=True):
        plt.show()