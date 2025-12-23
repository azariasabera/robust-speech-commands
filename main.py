import hydra
from omegaconf import DictConfig
from src.data.data_loader import download_dataset, load_datasets, get_class_labels, dataset_to_numpy
import numpy as np
from src.data.feature_extraction import extract_feature
from src.data.utils import cmvn_fit_train, cmvn_apply

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #download_dataset(cfg)
    ds_train, ds_val, ds_test, ds_info = load_datasets(cfg)
    label_names = get_class_labels(ds_info=ds_info)
    print(label_names)

    X_train, y_train = dataset_to_numpy(ds=ds_train, config=cfg)
    X_val, y_val = dataset_to_numpy(ds=ds_val, config=cfg)
    X_test, y_test = dataset_to_numpy(ds=ds_test, config=cfg)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    print(type(X_train[0][0]), type(y_train[0]))
    print(np.max(X_train[0]))
    print(np.min(X_train[0]))

    spec_train = extract_feature(waveforms=X_train, config=cfg)['mfcc']
    spec_val = extract_feature(waveforms=X_val, config=cfg)['mfcc']
    spec_test = extract_feature(waveforms=X_test, config=cfg)['mfcc']

    mean, std = cmvn_fit_train(Feat_train=spec_train)

    print(mean.shape, std.shape)

    spec_train_norm = cmvn_apply(Feat=spec_train, mean=mean, std=std)
    spec_val_norm = cmvn_apply(Feat=spec_val, mean=mean, std=std)
    spec_test_norm = cmvn_apply(Feat=spec_test, mean=mean, std=std)

    print(spec_train_norm.shape)
    print(spec_val_norm.shape)
    print(spec_test_norm.shape)

if __name__ == "__main__":
    main()