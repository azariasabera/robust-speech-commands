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