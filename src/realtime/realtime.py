# src/realtime/realtime.py

from omegaconf import DictConfig
from typing import List, Optional
import sounddevice as sd
import numpy as np
import time
from collections import Counter
import torch
from src.data.feature_extraction import extract_mfcc_fast
from src.data.utils import cmvn_apply
from src.realtime.utils import get_realtime_param, load_cmvn, load_realtime_model
from src.noise.denoise import wiener_filter, spectral_subtraction

def listen_and_detect(config: DictConfig, labels: List[str], model: Optional[torch.nn.Module] = None,
    mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> None:
    """
    Run a real-time keyword spotting loop, listening on the microphone and detecting commands.

    Loads the model and CMVN statistics if not provided, applies optional denoising, 
    extracts MFCC features, normalizes them, and performs inference using the model.
    Detected commands are printed in real-time with a configurable cooldown.

    Args:
        config (DictConfig): Hydra configuration object.
        labels (list[str]): List of class labels the model was trained on.
        model (Optional[torch.nn.Module]): Preloaded model for inference. If None, the model
            is loaded from default checkpoint paths in the config.
        mean (Optional[np.ndarray]): CMVN mean vector. If None, loaded from config paths.
        std (Optional[np.ndarray]): CMVN standard deviation vector. If None, loaded from config paths.
    """
    # Load params
    sr = get_realtime_param(config, "sr", 16000)
    buffer_sec = get_realtime_param(config=config, key="buffer_size", default=1.0)
    hop_sec = get_realtime_param(config=config, key="hop_size", default=0.1)
    cooldown = get_realtime_param(config=config, key="cooldown_time", default=10)
    use_filter = get_realtime_param(config=config, key="use_filter", default=False)
    filter_type = get_realtime_param(config=config, key="filter_type", default="weiner")
    selection = get_realtime_param(config=config, key="selection", default="majority_vote")

    if filter_type not in ["weiner", "spectral_subtraction"]:
        filter_type = "weiner"
    if selection not in ["majority_vote", "max_average_confidence"]:
        selection = "majority_vote"


    buffer_size = int(sr * buffer_sec)
    hop_size = int(sr * hop_sec)
    hop_ms = int(1000 * hop_sec) # since sleep() expects in ms

    # Load model + CMVN
    model, device = load_realtime_model(config=config, num_classes=len(labels), model=model)
    mean, std = load_cmvn(config=config, mean=mean, std=std)

    # Buffers
    buffer = np.zeros(buffer_size, dtype=np.float32)
    hop_predictions = []
    hop_probs = []

    last_trigger_time = 0.0
    last_command = None
    last_vote_time = time.time()

    # Audio callback
    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        buffer[:-frames] = buffer[frames:]
        buffer[-frames:] = indata[:, 0]

    stream = sd.InputStream(
        samplerate=sr,
        channels=1,
        blocksize=hop_size,
        callback=audio_callback,
    )

    print("Listening... Ctrl+C to stop")

    try:
        with stream:
            while True:
                audio_chunk = buffer.copy()

                if use_filter:
                    audio_chunk = (
                        wiener_filter(config=config, noisy_wave=audio_chunk, noise_wave=None) if filter_type=="weiner"
                        else spectral_subtraction(config=config, noisy_wave=audio_chunk, noise_wave=None)
                    )

                # MFCC
                mfcc_feat = extract_mfcc_fast(audio_chunk, config) # (F, T)
                mfcc_feat = mfcc_feat[np.newaxis, ...] # (1, F, T)

                # CMVN
                mfcc_norm = cmvn_apply(Feat=mfcc_feat, mean=mean, std=std)

                # Add channel dimension
                mfcc_norm = mfcc_norm[:, np.newaxis, :, :] # (1, 1, F, T)

                # Torch tensor
                mfcc_norm = torch.from_numpy(mfcc_norm).float().to(device)

                # Inference
                with torch.no_grad():
                    log_probs = model(mfcc_norm) # (1, num_classes)
                    probs = torch.exp(log_probs).cpu().numpy()[0]  # convert log probs to probs
                    pred = labels[np.argmax(probs)]

                hop_predictions.append(pred)
                hop_probs.append(probs)

                # 1s aggregation
                if time.time() - last_vote_time >= 1.0:
                    vote = (
                        majority_vote(hop_labels=hop_predictions) if selection=="majority_vote"
                        else max_confidence_vote(labels=labels, hop_probs=hop_probs)
                    )
                    hop_predictions.clear()
                    hop_probs.clear()
                    last_vote_time = time.time()

                    if vote not in {"_silence_", "_unknown_"}:
                        if ( time.time() - last_trigger_time >= cooldown or vote != last_command):
                            print(f"Detected command: {vote}")
                            last_trigger_time = time.time()
                            last_command = vote

                sd.sleep(hop_ms)

    except KeyboardInterrupt:
        print("Stopped listening.")

def majority_vote(hop_labels: List[str]) -> str:
    """
    Compute the majority vote from a list of predicted labels.

    Args:
        hop_labels (List[str]): List of predicted labels for a short audio hop.

    Returns:
        str: The label with the highest frequency in the hop_labels list.
    """
    return Counter(hop_labels).most_common(1)[0][0]

def max_confidence_vote(labels: List[str], hop_probs: List[np.ndarray]) -> str:
    """
    Compute the label with the maximum average confidence over a series of hops.

    Args:
        labels (List[str]): List of class labels corresponding to the model outputs.
        hop_probs (List[np.ndarray]): List of probability vectors from the model for each hop.
                    
    Returns:
        str: Label with the highest average predicted probability.
    """
    avg_probs = np.mean(hop_probs, axis=0)
    return labels[np.argmax(avg_probs)]
