# src/training/train_and_validate.py

import torch
import torch.nn as nn
from src.training.utils import get_device, get_training_param, set_seed
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig
from datetime import datetime

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


def train(config: DictConfig, model: nn.Module, train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader) -> Dict[str, list]:
    """
    Train a model with optional validation, early stopping, and model checkpointing.

    Args:
        config: Hydra DictConfig object.
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.

    Returns:
        Dictionary containing training history:
        'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """

    set_seed(config=config)

    device = get_device(config=config)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=get_training_param(config=config, key="learning_rate", default=1e-3),
        weight_decay=get_training_param(config=config, key="weight_decay", default=1e-4),
    )

    save = get_training_param(config=config, param="save", key="enabled", default=True)
    filename = get_training_param(config=config, param="save", key="filename", default="best_model.pt")
    save_dir = Path(get_training_param(config=config, param="save", key="dir", default=Path.cwd()/"saved_model"))
    save_dir.mkdir(parents=True, exist_ok=True)

    monitor = get_training_param(config=config, param="save", key="monitor", default="val_acc")
    assert monitor in ["val_acc", "val_loss"], "training.save.monitor must be 'val_acc' or 'val_loss'"
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_metric = float("-inf") if monitor == "val_acc" else float("inf") # We want to maximize acc but minimize loss

    curr = datetime.now().strftime("%Y%m%d_%H%M%S") # To avoid overwrite (per training run)
    save_pth = save_dir / f"{Path(filename).stem}_{curr}{Path(filename).suffix}"

    patience_counter = 0

    for epoch in range(get_training_param(config=config, key="epochs", default=100)):
        train_metrics = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = _validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_acc={train_metrics['acc']:.3f} "
            f"val_acc={val_metrics['acc']:.3f} | "
            f"train_loss={train_metrics['loss']:.3f} "
            f"val_loss={val_metrics['loss']:.3f}"
        )

        # Save best model
        monitor_value = val_metrics[monitor.replace("val_", "")]
        improved = (monitor_value > best_metric if monitor == "val_acc" else monitor_value < best_metric )

        if improved:
            best_metric = monitor_value
            patience_counter = 0
            if save:
                torch.save(model.state_dict(), save_pth)
        else:
            patience_counter += 1

        # Early stopping
        if get_training_param(config=config, param="early_stopping", key="enabled", default=True):
            if patience_counter >= get_training_param(config=config, param="early_stopping", key="patience", default=10):
                print(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(best {monitor}={best_metric:.4f})"
                )
                break

    return history