# src/training/train_and_validate.py

import torch
import torch.nn as nn
from typing import Dict

def _train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                    criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Dictionary with 'loss' and 'acc' for the epoch.
    """
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return {"loss": total_loss / total, "acc": total_correct / total}


@torch.no_grad()
def _validate_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader,
                        criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on validation/test data for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation/test data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Dictionary with 'loss' and 'acc' for the epoch.
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return {"loss": total_loss / total, "acc": total_correct / total}