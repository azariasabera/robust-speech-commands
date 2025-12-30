# src/realtime/realtime.py

from omegaconf import DictConfig
from typing import List
import sounddevice as sd
import numpy as np
import time
from collections import Counter
import torch
from src.data.feature_extraction import extract_mfcc_fast
from src.data.utils import cmvn_apply
from src.noise.denoise import wiener_filter, spectral_subtraction
from src.training.utils import get_device

def listen_and_detect(config: DictConfig, labels: list[str], model: torch.nn.Module, 
                      mean: np.ndarray, std: np.ndarray) -> None:
    """
    Run a real-time keyword spotting loop, listening on the microphone and detecting commands.

    Expects a preloaded model and CMVN statistics. Applies optional denoising, 
    extracts MFCC features, normalizes them, and performs inference using the model.
    Detected commands are printed in real-time with a configurable cooldown.

    Args:
        config (DictConfig): Hydra configuration object.
        labels (list[str]): List of class labels the model was trained on.
        model (torch.nn.Module): Model for inference.
        mean (np.ndarray): CMVN mean vector.
        std (np.ndarray): CMVN standard deviation vector.
    """
    # Load params from config
    sr = config.get("realtime", {}).get("sr", 16000)
    buffer_sec = config.get("realtime", {}).get("buffer_size", 1.0)
    hop_sec = config.get("realtime", {}).get("hop_size", 0.1)
    cooldown = config.get("realtime", {}).get("cooldown_time", 10)
    use_filter = config.get("realtime", {}).get("use_filter", False)
    filter_type = config.get("realtime", {}).get("filter_type", "wiener")
    selection = config.get("realtime", {}).get("selection", "majority_vote")
    device = get_device(config=config)

    if filter_type not in ["wiener", "spectral_subtraction"]:
        filter_type = "wiener"
    if selection not in ["majority_vote", "max_average_confidence"]:
        selection = "majority_vote"

    buffer_size = int(sr * buffer_sec)
    hop_size = int(sr * hop_sec)
    hop_ms = int(1000 * hop_sec)  # since sleep() expects in ms

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
                        wiener_filter(config=config, noisy_wave=audio_chunk, noise_wave=None)
                        if filter_type == "wiener"
                        else spectral_subtraction(config=config, noisy_wave=audio_chunk, noise_wave=None)
                    )
                # MFCC extraction
                mfcc_feat = extract_mfcc_fast(audio_chunk, config)[np.newaxis, ...]  # (1, F, T)
                mfcc_norm = cmvn_apply(Feat=mfcc_feat, mean=mean, std=std)
                mfcc_norm = mfcc_norm[:, np.newaxis, :, :]  # (1, 1, F, T) adds channel
                mfcc_norm_tensor = torch.from_numpy(mfcc_norm).float().to(device)

                # Inference
                with torch.no_grad():
                    log_probs = model(mfcc_norm_tensor) # (1, num_classes)
                    probs = torch.exp(log_probs).cpu().numpy()[0] # convert log probs to probs
                    pred = labels[np.argmax(probs)]

                hop_predictions.append(pred)
                hop_probs.append(probs)

                # 1s aggregation
                if time.time() - last_vote_time >= 1.0:
                    vote = (
                        majority_vote(hop_labels=hop_predictions)
                        if selection == "majority_vote"
                        else max_confidence_vote(labels, hop_probs)
                    )
                    hop_predictions.clear()
                    hop_probs.clear()
                    last_vote_time = time.time()
                    # Here I ignore unknown and silence from getting printed (can be configured in the future)
                    if vote not in {"_silence_", "_unknown_"}:
                        # Here I ignore printing same command twice (again can be configurable in the future)
                        if (time.time() - last_trigger_time >= cooldown) or vote != last_command: 
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