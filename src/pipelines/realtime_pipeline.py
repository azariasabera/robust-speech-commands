# src/pipelines/realtime_pipeline.py

from omegaconf import DictConfig
from src.pipelines.state import PipelineState, ModelSource
from src.realtime.realtime import listen_and_detect
from src.pipelines.utils import ensure_state


def run_realtime(config: DictConfig, state: PipelineState) -> None:
    """
    Start realtime keyword spotting.

    Uses the trained model and metadata from the current pipeline state
    when available; otherwise falls back to loading defaults from disk.

    Args:
        config (DictConfig): Hydra configuration object.
        state (PipelineState): Shared pipeline state containing the trained or
            loaded model, CMVN statistics, labels, and clean test data.
    """
    state = ensure_state(config, state, realtime=True)

    if state.model_source == ModelSource.TRAINED:
        print("[Realtime] Using currently trained model and CMVN ...")
    else:
        print("[Realtime] Using preloaded model and CMVN from default checkpoints ...")

    listen_and_detect(
        config=config,
        labels=state.labels,
        model=state.model,
        mean=state.mean,
        std=state.std,
    )