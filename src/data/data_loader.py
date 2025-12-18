# src/data/data_loader.py

from typing import Tuple, List
from omegaconf import DictConfig
import tensorflow_datasets as tfds
import tensorflow as tf

def download_dataset(config: DictConfig) -> None:
    """
    Download and prepare a TensorFlow Datasets dataset.

    This function triggers the download and preprocessing of the dataset
    and stores it in the specified data directory. If the dataset already
    exists in the cache, it will be reused.

    Args:
        config (DictConfig): Hydra configuration containing dataset parameters.
            Expected fields:
                - dataset.dataset_name (str)
                - dataset.root_dir (str or Path)
    """
    tfds.load(
        config.dataset.dataset_name,
        data_dir=config.dataset.root_dir,
        split=["train", "validation", "test"],
        with_info=False,
    )
    


def load_datasets( config: DictConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load train, validation, and test datasets from TensorFlow Datasets.

    Assumes the dataset has already been downloaded and prepared.
    If not found, raises a runtime error telling to download the dataset first.

    Args:
        config (DictConfig): Hydra configuration containing dataset parameters.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
            Train, validation, and test datasets.

    Raises:
        RuntimeError: If the dataset is not found locally.
    """
    try:
        ds_train, ds_val, ds_test = tfds.load(
            name=config.dataset.dataset_name,
            data_dir=config.dataset.root_dir,
            split=["train", "validation", "test"],
            as_supervised=True,
            shuffle_files=True,
            download=False,
        )
        return ds_train, ds_val, ds_test

    except (AssertionError): # tsds raises AssertionError
        raise RuntimeError(
            f"Dataset '{config.dataset.dataset_name}' was not found in "
            f"{config.dataset.root_dir}."
            "Please run the dataset download step first!"
        ) from None
