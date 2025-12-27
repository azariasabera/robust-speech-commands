# src/noise/utils.py

from omegaconf import DictConfig
from typing import List, Optional, Any
import librosa
import numpy as np
from pathlib import Path
import random
from src.data.utils import get_feature_param

def get_noise_param(config: DictConfig, param: Optional[str] = None, 
                    key: str = "", default: Any = None) -> Any:
    """
    Retrieve a parameter from config.noise with a safe fallback.

    Args:
        config: Hydra DictConfig object.
        param: Optional subsection under 'noise'.
        key: Parameter name to retrieve.
        default: Value to return if key is missing.

    Returns:
        The value from config.noise[key] or config.noise[param][key] or default.
    """
    noise_cfg = config.get("noise", {})

    if param:
        return noise_cfg.get(param, {}).get(key, default)

    return noise_cfg.get(key, default)


def load_and_trim_audio(config: DictConfig) -> List[np.ndarray]:
    """
    Load all .wav files in the noise folder specified in config,
    trim them into fixed-length segments (1 sec by default), 
    and discard any leftover audio shorter than 1 sec.

    Args:
        config: Hydra DictConfig object containing noise settings.

    Returns:
        List of 1-second audio segments as numpy arrays.
    
    Raises:
        FileNotFoundError: If the folder specified in config.noise.dir does not exist.
    """
    dir = Path(get_noise_param(config=config, key="dir", default=Path.cwd()/"data"/"noise"))
    folder_pth = dir / "musan" / "noise" / "free-sound"
    sr: int = get_noise_param(config=config, key="sr", default=16000)
    duration: float = get_noise_param(config=config, key="duration", default=1.0)

    if not folder_pth.exists():
        raise FileNotFoundError(f"Noise dataset folder does not exist: {folder_pth}")

    all_segments: List[np.ndarray] = []
    target_len: int = int(sr * duration)
    
    for wav_path in folder_pth.rglob("*.wav"):
        try:
            y, _ = librosa.load(wav_path, sr=sr)
        except Exception as e:
            print(f"Warning: failed to load {wav_path}: {e}")
            continue
        
        # Trim into 1-second segments
        num_segments = len(y) // target_len
        for i in range(num_segments):
            seg = y[i*target_len:(i+1)*target_len]
            all_segments.append(seg)
    
    return all_segments

def prepare_noise_for_test(
    config: DictConfig, 
    noise_segments: List[np.ndarray], 
    test_size: int
) -> np.ndarray:
    """
    Shuffle and select exactly `test_size` 1-second noise segments
    to match the test dataset.

    Args:
        config: Hydra DictConfig object containing noise settings.
        noise_segments: List of preprocessed 1-second noise segments.
        test_size: Number of noise samples needed (e.g., length of test set).

    Returns:
        numpy.ndarray: Array of shape (test_size, sr) with selected noise segments.
    
    Raises:
        ValueError: If there are not enough noise segments for the requested test size.
    """
    shuffle: bool = get_noise_param(config=config, key="shuffle", default=True)
    seed: int = get_noise_param(config=config, key="seed", default=42)

    random.seed(seed)
    np.random.seed(seed)
    
    if len(noise_segments) < test_size:
        raise ValueError(
            f"Not enough 1-sec noise segments ({len(noise_segments)}) for requested test size ({test_size})"
        )
    
    if shuffle:
        random.shuffle(noise_segments)
    
    selected = noise_segments[:test_size]
    
    return np.stack(selected, axis=0)

def stft(y: np.ndarray, config: DictConfig) -> np.ndarray:
    """
    Compute STFT of a single waveform.

    Args:
        y: 1D waveform array of shape (T,)
        config: Hydra config with STFT parameters

    Returns:
        Complex STFT matrix of shape (F, T_frames)
    """
    n_fft = int(get_feature_param(config=config, feature="stft", key="n_fft", default=1024))
    hop  = int(get_feature_param(config=config, feature="stft", key="hop_length", default=256))
    win  = int(get_feature_param(config=config, feature="stft", key="win_length", default=n_fft))
    window = str(get_feature_param(config=config, feature="stft", key="window", default="hann"))
    center = bool(get_feature_param(config=config, feature="stft", key="center", default=True))

    return librosa.stft( y, n_fft=n_fft, hop_length=hop, win_length=win, 
                        window=window, center=center, pad_mode="reflect")

def istft(Y: np.ndarray, config: DictConfig, length: Optional[int] = None) -> np.ndarray:
    """
    Compute the inverse Short-Time Fourier Transform (iSTFT).

    Args:
        Y: Complex STFT matrix of shape (F, T_frames), where:
           - F is the number of frequency bins
           - T_frames is the number of time frames
        config: Hydra DictConfig containing STFT parameters.
        length: Optional target length of the reconstructed waveform.
                If provided, the output is trimmed or zero-padded to
                exactly this length.

    Returns:
        np.ndarray:
            Reconstructed time-domain waveform as a 1D float32 array
            of shape (T,).
    """
    n_fft = int(get_feature_param(config=config, feature="stft", key="n_fft", default=1024))
    hop  = int(get_feature_param(config=config, feature="stft", key="hop_length", default=256))
    win  = int(get_feature_param(config=config, feature="stft", key="win_length", default=n_fft))
    window = str(get_feature_param(config=config, feature="stft", key="window", default="hann"))
    center = bool(get_feature_param(config=config, feature="stft", key="center", default=True))

    return librosa.istft(stft_matrix=Y, hop_length=hop, win_length=win, 
                         window=window, center=center, length=length)