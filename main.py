import hydra
from omegaconf import DictConfig
#from src.data.data_loader import download_dataset, load_datasets, get_class_labels, dataset_to_numpy
import numpy as np
#from src.data.feature_extraction import extract_feature
#from src.data.utils import cmvn_fit_train, cmvn_apply
import torch
from src.models.cnn import KeywordSpottingNet
from src.training.train_and_validate import train
from src.training.utils import get_device, plot_history, get_training_param
from src.evaluation.evaluate import evaluate
from torch.utils.data import TensorDataset, DataLoader

def load_data(config: DictConfig, X: np.ndarray, y: np.ndarray) -> DataLoader:
    """
    Convert numpy arrays to a PyTorch DataLoader.

    Args:
        config: Hydra DictConfig object.
        X: Input features, shape (N, H, W)
        y: Class labels, shape (N,)

    Returns:
        DataLoader yielding batches of (X, y) tensors suitable for CNN training.
    """
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    y_tensor = torch.from_numpy(y).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    batch_size = get_training_param(config=config, key="batch_size", default=64)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return loader

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """#download_dataset(cfg)
    ds_train, ds_val, ds_test, ds_info = load_datasets(cfg)
    label_names = get_class_labels(ds_info=ds_info)
    print(label_names)

    X_train, y_train = dataset_to_numpy(ds=ds_train, config=cfg)
    X_val, y_val = dataset_to_numpy(ds=ds_val, config=cfg)
    X_test, y_test = dataset_to_numpy(ds=ds_test, config=cfg)

    spec_train = extract_feature(waveforms=X_train, config=cfg)['mfcc']
    spec_val = extract_feature(waveforms=X_val, config=cfg)['mfcc']
    spec_test = extract_feature(waveforms=X_test, config=cfg)['mfcc']

    mean, std = cmvn_fit_train(Feat_train=spec_train)

    spec_train_norm = cmvn_apply(Feat=spec_train, mean=mean, std=std)
    spec_val_norm = cmvn_apply(Feat=spec_val, mean=mean, std=std)
    spec_test_norm = cmvn_apply(Feat=spec_test, mean=mean, std=std)"""

    # For test purposes only: I downloaded spec_*_norm as .npz in data/processed/
    tr = np.load("data/processed/train.npz")
    va = np.load("data/processed/val.npz")
    te = np.load("data/processed/test.npz")

    spec_train_norm = tr['X']
    spec_val_norm = va['X']
    spec_test_norm = te['X']

    y_tr = tr['y']
    y_va = va['y']
    y_te = te['y']

    train_loader = load_data(cfg, spec_train_norm, y_tr)
    val_loader = load_data(cfg, spec_val_norm, y_va)
    test_loader = load_data(cfg, spec_test_norm, y_te)

    device = get_device(config=cfg)

    model = KeywordSpottingNet(config=cfg, num_classes=12)

    history = train(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

    plot_history(history=history, config=cfg)

    res = evaluate(config=cfg, model=model, loader=test_loader, device=device)
    print('loss: ', res['loss'])
    print('acc: ', res['acc'])

if __name__ == "__main__":
    main()