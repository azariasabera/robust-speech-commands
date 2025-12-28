# src/pipelines/state.py

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import torch

@dataclass
class PipelineState:
    """
    Shared state container for pipeline execution.

    `PipelineState` holds models, normalization statistics, labels, and
    intermediate artifacts that are produced by one pipeline stage
    (e.g. training or noisy evaluation) and reused by subsequent stages
    (e.g. evaluation or realtime inference).

    This avoids reloading datasets, models, and statistics across pipeline
    while keeping pipeline dependencies explicit.
    """

    model: Optional[torch.nn.Module] = None
    device: Optional[torch.device] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    labels: Optional[list[str]] = None

    # evaluation artifacts
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    # noisy evaluation artifacts
    noisy_wavs: Optional[Dict[str, np.ndarray]] = None
    noise_segments: Optional[np.ndarray] = None
