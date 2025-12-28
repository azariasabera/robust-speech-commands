# src/data/utils.py

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Tuple
import numpy as np
from pathlib import Path
import tensorflow_datasets as tfds

def get_feature_param(config: DictConfig, feature: str, key: str, default: Any) -> Any:
    """
    Get a parameter from config.features.<feature> with a fallback default.

    Args:
        config: Hydra DictConfig object.
        feature: Feature name (e.g., 'mfcc', 'mel')
        key: Parameter name inside the feature
        default: Value to return if key is missing

    Returns:
        The value from config.features[feature][key] or default
    """

    return config.get("features", {}).get(feature, {}).get(key, default)

def pad_or_trim_tensor(audio: tf.Tensor, config: DictConfig) -> tf.Tensor:
    """
    Pad or trim a TensorFlow audio tensor to a fixed duration using parameters from config.audio.

    Args:
        audio: 1D tf.Tensor containing the audio waveform.
        config: Hydra DictConfig object containing audio parameters (`sr`, `duration`).

    Returns:
        1D tf.Tensor of length `sr * duration`, either padded with zeros or trimmed.
    """
    sr = getattr(config.audio, "sr", 16000)
    duration = getattr(config.audio, "duration", 1.0)  # default 1 second
    target_len = int(sr * duration)

    audio_len = tf.shape(audio)[0]

    audio = tf.cond(
        audio_len > target_len,
        lambda: audio[:target_len],
        lambda: tf.pad(audio, [[0, target_len - audio_len]])
    )
    return audio

def cmvn_fit_train(Feat_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation for Cepstral Mean & Variance Normalization (CMVN).

    Args:
        Feat_train (np.ndarray): Training features of shape (num_samples, F, n_frames). F is n_mel or n_mfcc for e.g.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing
            - mean: Mean over samples and frames, shape (1, F, 1)
            - std: Standard deviation over samples and frames, shape (1, F, 1)
    """
    mean = Feat_train.mean(axis=(0, 2), keepdims=True)
    std  = Feat_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std


def cmvn_apply(Feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply Cepstral Mean & Variance Normalization (CMVN) to features.

    Args:
        Feat (np.ndarray): Features to normalize, shape (num_samples, F, n_frames)
        mean (np.ndarray): Mean computed from training data, shape (1, F, 1)
        std (np.ndarray): Standard deviation computed from training data, shape (1, F, 1)

    Returns:
        np.ndarray: Normalized features of the same shape as input X
    """
    return (Feat - mean) / std

def dataset_exists(config: DictConfig) -> bool:
    """
    Check whether the specified TFDS dataset has already been downloaded.

    Looks in the configured 'root_dir' for the dataset folder corresponding to the
    dataset name and version. Returns True if the folder exists and contains files,
    indicating the dataset is ready for use.

    Args:
        config (DictConfig): Hydra configuration object containing:
            - dataset.dataset_name: Name of the TFDS dataset.
            - dataset.root_dir: Root directory where datasets are stored.

    Returns:
        bool: True if the dataset exists and is non-empty, False otherwise.
    """
    
    builder = tfds.builder(config.dataset.dataset_name)
    dataset_dir = Path(config.dataset.root_dir) / builder.name / str(builder.version)
    return dataset_dir.exists() and any(dataset_dir.iterdir())  # folder is not empty