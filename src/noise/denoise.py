# src/noise/denoise.py

import numpy as np
from omegaconf import DictConfig
from src.noise.utils import get_noise_param, stft, istft

def _estimate_noise_psd_energy_vad(
    noisy_wave: np.ndarray,
    config: DictConfig,
) -> np.ndarray:
    """
    Estimate noise Power Spectral Density (PSD) from a noisy waveform
    using an energy-based voice activity detection (VAD).

    The method assumes that low-energy frames are dominated by noise.

    Steps:
        1. Compute STFT of the noisy signal.
        2. Compute per-frame energy (mean power across frequency bins).
        3. Select low-energy frames as noise-only frames.
        4. Average their power spectra to estimate noise PSD.

    Args:
        noisy_wave: 1D numpy array of shape (T,)
        config: Hydra DictConfig with noise parameters

    Returns:
        Noise PSD estimate of shape (F, 1)
    """
    eps = float(get_noise_param(config, key="epsilon", default=1e-8))
    rms_th = float(get_noise_param(config, key="rms_threshold", default=1e-4))
    apply_th = bool(get_noise_param(config, key="apply_threshold", default=True))
    perc = float(get_noise_param(config, key="threshold_percentile", default=20.0))
    hang = int(get_noise_param(config, key="hangover_frames", default=0))

    Y = stft(y=noisy_wave, config=config)                      # (F, T)
    P = np.abs(Y)**2                                # power (F, T)
    frame_energy = P.mean(axis=0)                   # (T,)
    # Dynamic threshold: low percentile of energies
    dyn_th = np.percentile(frame_energy, perc) if apply_th else rms_th
    th = max(rms_th, dyn_th) if apply_th else rms_th

    noise_mask = frame_energy <= th                 # (T,) boolean

    # Optional simple smoothing (hangover): extend True regions by 'hang' frames
    if hang > 0:
        sm = noise_mask.astype(np.int32)
        for t in range(1, len(sm)):
            if sm[t] == 1 and sm[t-1] == 0:
                # back-fill previous 'hang' frames as noise (soft heuristic)
                start = max(0, t - hang)
                sm[start:t] = 1
        noise_mask = sm.astype(bool)

    # If no frames detected, fall back to min-statistics
    if not noise_mask.any():
        print("Changing to min_stats estimation")
        return _estimate_noise_psd_min_stats(noisy_wave, config)

    # Average over noise-only frames -> PSD per bin (F, 1)
    N_psd = P[:, noise_mask].mean(axis=1, keepdims=True) + eps
    return N_psd


def _estimate_noise_psd_min_stats(
    noisy_wave: np.ndarray,
    config: DictConfig,
) -> np.ndarray:
    """
    Estimate the noise Power Spectral Density (PSD) using minimum statistics.

    This method assumes that, for each frequency bin, the noise power
    corresponds to a low percentile of the observed power over time.
    It is a simple and robust blind noise estimation technique that does
    not rely on explicit voice activity detection.

    Procedure:
        1. Compute the STFT of the noisy waveform.
        2. Compute the power spectrogram |Y|^2 with shape (F, T).
        3. For each frequency bin, take a low percentile (e.g., 10th)
           across time frames to approximate the noise floor.

    Args:
        noisy_wave: 1D numpy array of shape (T,) containing the noisy waveform.
        config: Hydra DictConfig containing noise and STFT parameters.

    Returns:
        np.ndarray:
            Estimated noise PSD of shape (F, 1), where F is the number of
            frequency bins.
    """
    eps = float(get_noise_param(config, key="epsilon", default=1e-8))
    pctl = float(get_noise_param(config, key="min_stats_percentile", default=10.0))

    Y = stft(y=np.asarray(noisy_wave, dtype=np.float32), config=config)
    P = np.abs(Y)**2
    N_psd = np.percentile(P, pctl, axis=1, keepdims=True) + eps   # (F, 1)
    return N_psd


def _estimate_noise_psd_known_noise(
    noise_wave: np.ndarray,
    config: DictConfig,
) -> np.ndarray:
    """
    Estimate the noise Power Spectral Density (PSD) using a known noise signal.

    This method assumes that a clean noise-only waveform corresponding to
    the mixture is available. The noise PSD is computed directly from the
    STFT of this noise segment.

    Procedure:
        1. Compute the STFT of the noise-only waveform.
        2. Compute the power spectrum |N|^2.
        3. Average the power across time frames to obtain a per-frequency
           noise PSD estimate.

    Args:
        noise_wave: 1D numpy array of shape (T,) containing a noise-only waveform.
        config: Hydra DictConfig containing STFT and noise parameters.

    Returns:
        np.ndarray:
            Estimated noise PSD of shape (F, 1), where F is the number of
            frequency bins.
    """
    eps = float(get_noise_param(config, key="epsilon", default=1e-8))
    N = stft(y=np.asarray(noise_wave, dtype=np.float32), config=config)
    N_psd = (np.abs(N)**2).mean(axis=1, keepdims=True) + eps       # (F, 1)
    return N_psd