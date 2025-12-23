# src/data/data_loader.py

from typing import Tuple, List
from omegaconf import DictConfig
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src.data.utils import pad_or_trim_tensor

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
        name=config.dataset.dataset_name,
        data_dir=config.dataset.root_dir,
        split=["train", "validation", "test"],
        with_info=False,
    )
    


def load_datasets( config: DictConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load train, validation, and test datasets from TensorFlow Datasets.

    Each dataset yields tuples (audio, label) because `as_supervised=True`:
        - audio: tf.Tensor of shape (samples,) representing raw waveform
        - label: tf.Tensor scalar containing class index

    Args:
        config (DictConfig): Hydra configuration containing dataset parameters.
            Expected fields:
                - dataset.dataset_name (str)
                - dataset.root_dir (str or Path)

    Returns:
        Tuple containing:
            - ds_train (tf.data.Dataset): Training dataset
            - ds_val (tf.data.Dataset): Validation dataset
            - ds_test (tf.data.Dataset): Test dataset
            - ds_info (tfds.core.DatasetInfo): Metadata for the dataset, including class labels

    Raises:
        RuntimeError: If the dataset is not found locally.
    """
    try:
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            name=config.dataset.dataset_name,
            data_dir=config.dataset.root_dir,
            split=["train", "validation", "test"],
            as_supervised=True,
            shuffle_files=True,
            with_info=True,
            download=False,
        )
        return ds_train, ds_val, ds_test, ds_info

    except (AssertionError): # tsds raises AssertionError
        raise RuntimeError(
            f"Dataset '{config.dataset.dataset_name}' was not found in "
            f"{config.dataset.root_dir}. "
            "Please run the dataset download step first!"
        ) from None

def get_class_labels(ds_info: tfds.core.DatasetInfo) -> List[str]:
    """
    Return the list of class labels for a TFDS dataset.

    Args:
        ds_info (tfds.core.DatasetInfo): Dataset metadata returned by `load_datasets`.

    Returns:
        List[str]: Ordered list of label names, where index corresponds to
                   the integer label in supervised datasets.
    
    Example:
        labels = get_class_labels(ds_info)
        # labels[0] might be 'yes', labels[1] -> 'no', etc.
    """
    return ds_info.features["label"].names


def dataset_to_numpy(ds: tf.data.Dataset, config: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert TFDS dataset to NumPy arrays with padding/trimming and batch-wise iteration.

    Args:
        ds: tf.data.Dataset yielding (audio, label)
        config: DictConfig containing audio parameters

    Returns:
        X: np.ndarray of shape (num_samples, fixed_length)
        y: np.ndarray of shape (num_samples,)
    """
    batch_size = getattr(config, "batch_size", 64) # default batch_size

    ds = ds.map(lambda x, y: (pad_or_trim_tensor(x, config), y))
    ds = ds.batch(batch_size)

    X_list = []
    y_list = []
    
    for batch_audio, batch_label in tqdm(tfds.as_numpy(ds), desc="Converting ds to np"):
        X_list.append(batch_audio)
        y_list.append(batch_label)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y
