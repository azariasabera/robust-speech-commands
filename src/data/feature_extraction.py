
# src/data/feature_extraction.py

from typing import Dict, List
import numpy as np
import librosa
from omegaconf import DictConfig
from tqdm import tqdm
from src.data.utils import get_feature_param

def extract_stft( waveforms: List[np.ndarray], config: DictConfig) -> Dict[str, np.ndarray]:
    """
    Extracts Short-Time Fourier Transform (STFT) features from a list of waveforms.

    Args:
        waveforms (List[np.ndarray]): List of 1D NumPy arrays containing audio signals.
        config (DictConfig): Configuration containing STFT parameters.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - "stft_mag": Magnitude spectrograms, shape (N, F, T)
            - "stft_db": Decibel-scaled spectrograms, shape (N, F, T)
    """

    n_fft = int(get_feature_param(config, "stft", "n_fft", 1024))
    win_length = int(get_feature_param(config, "stft", "win_length", n_fft))
    hop_length = int(get_feature_param(config, "stft", "hop_length", 256))
    window = str(get_feature_param(config, "stft", "window", "hann"))
    center = bool(get_feature_param(config, "stft", "center", True))

    mag_specs, db_specs = [], []
    for waveform in tqdm(waveforms, desc="STFT"):
        spec = librosa.stft(
            y=waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode="reflect",
        )
        spec_mag = np.abs(spec).astype(np.float32)
        spec_db = librosa.amplitude_to_db(spec_mag, ref=np.max).astype(np.float32)
        mag_specs.append(spec_mag)
        db_specs.append(spec_db)

    return {
        "stft_mag": np.array(mag_specs, dtype=np.float32),  # (N, F, T)
        "stft_db": np.array(db_specs, dtype=np.float32),    # (N, F, T)
    }


def extract_mel(waveforms: List[np.ndarray], config: DictConfig) -> np.ndarray:
    """
    Extracts Mel-spectrogram features from a list of waveforms.

    Args:
        waveforms (List[np.ndarray]): List of 1D NumPy arrays containing audio signals.
        config (DictConfig): Configuration containing Mel-spectrogram parameters.

    Returns:
        np.ndarray: Mel-spectrograms, shape (N, n_mels, T)
    """

    sr = int(config.get("audio", {}).get("sr", 16000))
    n_fft = int(get_feature_param(config, "mel", "n_fft", 1024))
    win_length = int(get_feature_param(config, "mel", "win_length", n_fft))
    hop_length = int(get_feature_param(config, "mel", "hop_length", 256))
    n_mels = int(get_feature_param(config, "mel", "n_mels", 64))
    window = str(get_feature_param(config, "mel", "window", "hann"))
    center = bool(get_feature_param(config, "mel", "center", True))
    fmin = float(get_feature_param(config, "mel", "fmin", 0.0))
    fmax = float(get_feature_param(config, "mel", "fmax", None))

    to_db = bool(get_feature_param(config, "settings", "mel_to_db", True))

    mel_specs = []
    for waveform in tqdm(waveforms, desc=f"Mel({'dB' if to_db else 'power'})"):
        spec_mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            power=2.0, # power spectrogram (magnitude^2)
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        ).astype(np.float32)
        if to_db:
            spec_mel = librosa.power_to_db(spec_mel, ref=np.max).astype(np.float32)
        mel_specs.append(spec_mel)

    return np.array(mel_specs, dtype=np.float32)  # (N, n_mels, T)

def extract_mfcc(waveforms: List[np.ndarray], config: DictConfig) -> np.ndarray:
    """
    Extracts MFCC features (optionally with delta and delta-delta) from waveforms.

    Args:
        waveforms (List[np.ndarray]): List of 1D NumPy arrays containing audio signals.
        config (DictConfig): Configuration containing MFCC parameters.

    Returns:
        np.ndarray: MFCC features, shape (N, F, T). F = n_mfcc*3 if deltas included.
    """

    sr = int(config.get("audio", {}).get("sr", 16000))
    n_fft = int(get_feature_param(config, "mfcc", "n_fft", 1024))
    win_length = int(get_feature_param(config, "mfcc", "win_length", n_fft))
    hop_length = int(get_feature_param(config, "mfcc", "hop_length", 256))
    n_mels = int(get_feature_param(config, "mfcc", "n_mels", 64))
    n_mfcc = int(get_feature_param(config, "mfcc", "n_mfcc", 20))
    window = str(get_feature_param(config, "mfcc", "window", "hann"))
    center = bool(get_feature_param(config, "mfcc", "center", True))
    fmin = float(get_feature_param(config, "mfcc", "fmin", 0.0))
    fmax = float(get_feature_param(config, "mfcc", "fmax", None))

    include_deltas = bool(get_feature_param(config, "settings", "mfcc_include_deltas", True))

    feats = []
    for waveform in tqdm(waveforms, desc=f"MFCC{'(+Δ+ΔΔ)' if include_deltas else ''}"):
        
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        ).astype(np.float32)

        if include_deltas:
            delta = librosa.feature.delta(mfcc, order=1).astype(np.float32)
            delta2 = librosa.feature.delta(mfcc, order=2).astype(np.float32)
            # Stack along feature axis: shape becomes (n_mfcc*3, T)
            mfcc = np.concatenate([mfcc, delta, delta2], axis=0)

        feats.append(mfcc)

    return np.array(feats, dtype=np.float32)  # (N, F, T)

def extract_all_features(waveforms: List[np.ndarray], config: DictConfig) -> Dict[str, np.ndarray]:
    """
    Extracts STFT, Mel-spectrogram, and MFCC features from a list of waveforms.

    Args:
        waveforms (List[np.ndarray]): List of 1D NumPy arrays containing audio signals.
        config (DictConfig): Configuration containing all feature extraction parameters.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing all features:
            - "stft_mag": STFT magnitude, shape (N, F, T)
            - "stft_db": STFT decibel, shape (N, F, T)
            - "mel_db": Mel-spectrogram in dB, shape (N, n_mels, T)
            - "mfcc": MFCC features, shape (N, F, T)
    """
    
    stft_feats = extract_stft(waveforms, config)
    mel_db = extract_mel(waveforms, config)
    mfcc = extract_mfcc(waveforms, config)

    return {
        "stft_mag": stft_feats["stft_mag"],  # (N, F, T)
        "stft_db": stft_feats["stft_db"],    # (N, F, T)
        "mel_db": mel_db,                    # (N, n_mels, T)
        "mfcc": mfcc,                        # (N, F, T) [or F=3*n_mfcc if deltas included]
    }
