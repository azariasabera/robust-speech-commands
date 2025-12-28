# src/pipelines/train_pipeline.py

from omegaconf import DictConfig

from src.pipelines.state import PipelineState
from src.data.utils import dataset_exists, cmvn_fit_train, cmvn_apply
from src.data.data_loader import (
    download_dataset,
    load_datasets,
    get_class_labels,
    dataset_to_numpy,
)
from src.data.feature_extraction import extract_feature
from src.training.utils import load_data, get_device, plot_history
from src.training.train_and_validate import train
from src.models.cnn import KeywordSpottingNet


def run_training(config: DictConfig, state: PipelineState) -> PipelineState:
    """
    Train the keyword spotting model and populate shared pipeline state.

    Args:
        config: Hydra configuration.
        state: PipelineState to be populated.

    Returns:
        Updated PipelineState with trained model, CMVN stats, labels, and test data.
    """
    if not dataset_exists(config):
        download_dataset(config)

    ds_train, ds_val, ds_test, ds_info = load_datasets(config)
    labels = get_class_labels(ds_info)

    X_train, y_train = dataset_to_numpy(ds_train, config)
    X_val, y_val = dataset_to_numpy(ds_val, config)
    X_test, y_test = dataset_to_numpy(ds_test, config)

    spec_train = extract_feature(X_train, config)["mfcc"]
    spec_val = extract_feature(X_val, config)["mfcc"]
    spec_test = extract_feature(X_test, config)["mfcc"]

    mean, std = cmvn_fit_train(spec_train)

    spec_train = cmvn_apply(spec_train, mean, std)
    spec_val = cmvn_apply(spec_val, mean, std)
    spec_test = cmvn_apply(spec_test, mean, std)

    train_loader = load_data(config, spec_train, y_train)
    val_loader = load_data(config, spec_val, y_val)

    device = get_device(config)
    model = KeywordSpottingNet(config, num_classes=len(labels)).to(device)

    history = train(config, model, train_loader, val_loader)
    plot_history(history, config)

    # populate state
    state.model = model
    state.device = device
    state.mean = mean
    state.std = std
    state.labels = labels
    state.X_test = X_test
    state.y_test = y_test

    return state