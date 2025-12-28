# src/pipelines/evaluate_pipeline.py

from omegaconf import DictConfig
from src.pipelines.state import PipelineState
from src.data.utils import dataset_exists, cmvn_apply
from src.data.data_loader import (
    download_dataset,
    load_datasets,
    get_class_labels,
    dataset_to_numpy,
)
from src.data.feature_extraction import extract_feature
from src.training.utils import load_data
from src.evaluation.evaluate import evaluate
from src.realtime.utils import load_realtime_model, load_cmvn
from src.noise.utils import load_and_trim_audio, prepare_noise_for_test, musan_exists
from src.noise.add_noise import add_noise_to_waveforms
from src.noise.download_dataset import download_musan


def run_evaluate_clean(config: DictConfig, state: PipelineState) -> None:
    """
    Evaluate the model on clean test data.

    If training has already been executed in the current run, this function
    reuses the model, device, CMVN statistics, and test data stored in
    'PipelineState'. Otherwise, it loads the required model, normalization
    statistics, labels, and test dataset from disk.

    Args:
        config (DictConfig): Hydra configuration object.
        state (PipelineState): Shared pipeline state containing training
            artifacts and evaluation data.
    """

    if state.model is None:
        if not dataset_exists(config):
            download_dataset(config)

        _, _, ds_test, ds_info = load_datasets(config)
        labels = get_class_labels(ds_info)
        X_test, y_test = dataset_to_numpy(ds_test, config)

        spec = extract_feature(X_test, config)["mfcc"]
        mean, std = load_cmvn(config)
        spec = cmvn_apply(spec, mean, std)

        loader = load_data(config, spec, y_test)
        model, device = load_realtime_model(config, num_classes=len(labels))
    else:
        spec = extract_feature(state.X_test, config)["mfcc"]
        spec = cmvn_apply(spec, state.mean, state.std)
        loader = load_data(config, spec, state.y_test)
        model, device = state.model, state.device

    res = evaluate(config, model, loader, device)
    print(res)


def run_evaluate_noisy(config: DictConfig, state: PipelineState) -> PipelineState:
    """
    Evaluate the model on noisy test data.

    This function adds noise to the clean test waveforms at multiple SNR levels,
    evaluates the model on each noisy variant, and stores the generated noisy
    waveforms and corresponding noise segments in the shared pipeline state.
    These artifacts are required for subsequent denoised evaluation.

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

    if state.X_test is None:
        raise RuntimeError("No test data available. Run train or clean evaluation first.")

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