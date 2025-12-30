# main.py

import hydra
from omegaconf import DictConfig

from src.pipelines.state import PipelineState
from src.pipelines.train_pipeline import run_training
from src.pipelines.evaluate_pipeline import (
    run_evaluate_clean,
    run_evaluate_noisy,
    run_evaluate_denoised,
)
from src.pipelines.realtime_pipeline import run_realtime


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Entry point for the keyword spotting pipeline.

    This function orchestrates training, evaluation, and realtime inference
    based on the selected pipeline in the configuration. A shared PipelineState
    object is passed between pipeline stages to reuse models, statistics,
    datasets, and intermediate artifacts when available.
    
    Args:
        cfg (DictConfig): Hydra configuration object controlling pipeline,
            paths, model parameters, and runtime options.
    """
     
    pipeline = list(cfg.get("pipeline", ["realtime"]))
    state = PipelineState()

    if "train" in pipeline:
        state = run_training(cfg, state)

    if "evaluate_clean" in pipeline:
        state = run_evaluate_clean(cfg, state)

    if "evaluate_noisy" in pipeline:
        state = run_evaluate_noisy(cfg, state)

    if "evaluate_denoised" in pipeline:
        run_evaluate_denoised(cfg, state)

    if "realtime" in pipeline:
        run_realtime(cfg, state)


if __name__ == "__main__":
    main()