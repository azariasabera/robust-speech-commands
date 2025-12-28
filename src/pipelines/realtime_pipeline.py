# src/pipelines/realtime_pipeline.py

from omegaconf import DictConfig
from src.pipelines.state import PipelineState
from src.data.utils import dataset_exists
from src.data.data_loader import download_dataset, load_datasets, get_class_labels
from src.realtime.realtime import listen_and_detect
from src.realtime.utils import get_realtime_param


def run_realtime(config: DictConfig, state: PipelineState) -> None:
    """
    Start realtime keyword spotting.
    Uses trained model if available, otherwise loads defaults.

    Args:
        config (DictConfig): Hydra configuration object.
        state (PipelineState): Shared pipeline state containing the trained or
            loaded model, CMVN statistics, labels, and clean test data.
    """
    if not dataset_exists(config):
        download_dataset(config)

    _, _, _, ds_info = load_datasets(config)
    labels = get_class_labels(ds_info)

    # If there is a new trained model and if 'use_default' param is False
    if state.model is not None and not get_realtime_param(config, "use_default", False):
        listen_and_detect(
            config=config,
            labels=labels,
            model=state.model,
            mean=state.mean,
            std=state.std,
        )
    else:
        listen_and_detect(config=config, labels=labels)