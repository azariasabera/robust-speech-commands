# src/noise/add_noise.py

import numpy as np
from typing import Dict, List
from omegaconf import DictConfig
from src.noise.utils import get_noise_param

def add_noise_to_waveforms(
    config: DictConfig,
    clean_waves: np.ndarray,
    noise_waves: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Add MUSAN noise to clean waveforms at multiple controlled SNR levels.

    Noise is added sample-wise (1-to-1): the i-th noise waveform is added
    to the i-th clean waveform. This assumes noise_waves has already been
    shuffled and trimmed to match the test set size in 'prepare_noise_for_test'.

    Args:
        config: Hydra DictConfig object containing noise parameters.
        clean_waves: Clean audio waveforms, shape (N, T).
        noise_waves: Noise waveforms, shape (N, T).

    Returns:
        Dict[int, np.ndarray]:
            Dictionary mapping SNR value (in dB) to noisy waveforms
            of shape (N, T).
    """
    snr_list: List[int] = list(
        get_noise_param(config=config, key="snr", default=[0, 5, 10, 20])
    )
    eps: float = float(get_noise_param(config=config, key="epsilon", default=1e-8))
    clip: bool = bool(get_noise_param(config=config, key="clip", default=True))
    rms_threshold: float = float(
        get_noise_param(config=config, key="rms_threshold", default=1e-4)
    )
    apply_threshold: bool = bool(
        get_noise_param(config=config, key="apply_threshold", default=True)
    )

    assert clean_waves.ndim == 2, "clean_waves must have shape (N, T)"
    assert noise_waves.ndim == 2, "noise_waves must have shape (N, T)"
    assert clean_waves.shape == noise_waves.shape, (
        "clean_waves and noise_waves must have identical shape"
    )

    noisy_versions: Dict[int, np.ndarray] = {}

    for snr_db in snr_list:
        noisy_waves = np.empty_like(clean_waves)

        for i in range(clean_waves.shape[0]):
            wav = clean_waves[i]
            noise = noise_waves[i]

            # RMS energy
            rms_wav = np.sqrt(np.mean(wav ** 2) + eps)
            rms_noise = np.sqrt(np.mean(noise ** 2) + eps)

            # If RMS energy is too low, e.g., silence
            if apply_threshold and rms_wav < rms_threshold:
                noisy_waves[i] = wav
                continue

            # Scale noise to achieve desired SNR
            scale = rms_wav / (rms_noise * (10 ** (snr_db / 20)))
            noisy = wav + scale * noise

            if clip:
                noisy = np.clip(noisy, -1.0, 1.0)

            noisy_waves[i] = noisy

        noisy_versions[snr_db] = noisy_waves

    return noisy_versions