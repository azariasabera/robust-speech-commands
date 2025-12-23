# src/data/utils.py

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any

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