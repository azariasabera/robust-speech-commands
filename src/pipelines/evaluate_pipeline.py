# src/pipelines/evaluate_pipeline.py

from omegaconf import DictConfig
from src.pipelines.state import PipelineState
from src.data.utils import cmvn_apply
from src.data.feature_extraction import extract_feature
from src.training.utils import load_data
from src.evaluation.evaluate import evaluate
from src.noise.utils import load_and_trim_audio, prepare_noise_for_test, musan_exists
from src.noise.add_noise import add_noise_to_waveforms
from src.noise.download_dataset import download_musan
from src.noise.denoise import wiener_filter, spectral_subtraction
from src.pipelines.utils import ensure_state


def run_evaluate_clean(config: DictConfig, state: PipelineState) -> PipelineState:
    """
    Evaluate the model on clean test data.

    This function evaluates the keyword spotting model on clean test samples.
    If training has already been executed in the current run, the model,
    CMVN statistics, labels, and test data stored in `PipelineState` are reused.

    If these artifacts are not present, they are loaded from disk using the
    configured default checkpoints and statistics, and stored in the shared
    pipeline state for reuse by subsequent pipeline stages.

    Args:
        config (DictConfig): Hydra configuration object.
        state (PipelineState): Shared pipeline state containing training
            artifacts and evaluation data.

    Returns:
        PipelineState: Updated pipeline state containing model, device, CMVN
        statistics, labels, and test data.
    """

    state = ensure_state(config=config, state=state)

    print(f"[Eval] Using {state.model_source.value} model and CMVN ...")

    spec = extract_feature(state.X_test, config)["mfcc"]
    spec = cmvn_apply(spec, state.mean, state.std)
    loader = load_data(config, spec, state.y_test)
    model, device = state.model, state.device

    res = evaluate(config, model, loader, device)
    print(res)
    return state


def run_evaluate_noisy(config: DictConfig, state: PipelineState) -> PipelineState:
    """
    Evaluate the model on noisy test data.

    This function adds noise to clean test waveforms at multiple SNR levels and
    evaluates the model on each noisy variant. If required artifacts (model,
    CMVN statistics, or test data) are missing from the pipeline state, they are
    loaded from disk using the configured defaults.

    The generated noisy waveforms and corresponding noise segments are stored
    in the shared pipeline state for use in subsequent denoised evaluation.

    Args:
        config (DictConfig): Hydra configuration object.
        state (PipelineState): Shared pipeline state containing the trained or
            loaded model, CMVN statistics, labels, and clean test data.

    Returns:
        PipelineState: Updated pipeline state with noisy evaluation artifacts
        ('noisy_wavs' and 'noise_segments') populated.
    """

    if not musan_exists(config):
        download_musan(config)
    
    state = ensure_state(config=config, state=state)

    print(f"[Eval] Using {state.model_source.value} model and CMVN ...")

    all_seg = load_and_trim_audio(config)
    selected = prepare_noise_for_test(config, all_seg, test_size=state.X_test.shape[0])
    noisy_wavs = add_noise_to_waveforms(config, state.X_test, selected)

    for snr, wav in noisy_wavs.items():
        spec = extract_feature(wav, config)["mfcc"]
        spec = cmvn_apply(spec, state.mean, state.std)
        loader = load_data(config, spec, state.y_test)

        res = evaluate(config, state.model, loader, state.device)
        print(f"SNR {snr}: {res}")

    state.noisy_wavs = noisy_wavs
    state.noise_segments = selected
    return state


def run_evaluate_denoised(config: DictConfig, state: PipelineState) -> None:
    """
    Evaluate model on denoised noisy signals.
    
    Note:
        This function requires `run_evaluate_noisy` to be executed first.
        The shared PipelineState must contain:
          - state.noisy_wavs
          - state.noise_segments
    
    Args:
        config (DictConfig): Hydra configuration object.
        state (PipelineState): Shared pipeline state containing the trained or
            loaded model, CMVN statistics, labels, and clean test data.

    Raises:
        RuntimeError: If noisy evaluation artifacts are missing.
    """
    if state.noisy_wavs is None or state.noise_segments is None:
        raise RuntimeError("Denoised evaluation requires noisy evaluation first.")

    for i, (snr, wav) in enumerate(state.noisy_wavs.items()):
        den_wiener = wiener_filter(config, wav, state.noise_segments[i])
        den_ss = spectral_subtraction(config, wav, state.noise_segments[i])

        for name, den in [("wiener", den_wiener), ("ss", den_ss)]:
            spec = extract_feature(den, config)["mfcc"]
            spec = cmvn_apply(spec, state.mean, state.std)
            loader = load_data(config, spec, state.y_test)

            res = evaluate(config, state.model, loader, state.device)
            print(f"{name} @ {snr}: {res}")