# src/evaluation/evaluate.py

import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm
from omegaconf import DictConfig

from src.evaluation.utils import plot_confusion

@torch.no_grad()
def evaluate(config: DictConfig, model: nn.Module, loader: torch.utils.data.DataLoader, 
             device: torch.device) -> Dict[str, float]:
    """
    Evaluate a trained model on a dataset and save a confusion matrix.
    
    Args:
        model (nn.Module): Trained PyTorch model (already moved to `device`).
        loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device on which tensors are evaluated.
        config (DictConfig, optional): Hydra config object for plotting settings.

    Returns:
        Dict[str, float]: Dictionary containing:
            - "loss": Average loss over the dataset
            - "acc": Classification accuracy over the dataset
    """
    model.eval()
    criterion = nn.CrossEntropyLoss() # I do it manually, could be improved

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    for X, y in tqdm(loader, desc="Evaluating: "):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    plot_confusion(config=config, y_true=all_targets, y_pred=all_preds)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }