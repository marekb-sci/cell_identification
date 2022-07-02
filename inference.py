# -*- coding: utf-8 -*-

import torch
# import pytorch_lightning as pl

from serialize import load_json
from data import NumpyCropToTensor, get_augmentation_tv
from training import ParticlesClassifier

def get_model(checkpoint_path, config):
    return ParticlesClassifier.load_from_checkpoint(checkpoint_path, config=config)

def get_image_transform(config):
    data_config = config['data']
    transform_test = get_augmentation_tv(data_config['transform_test']+data_config['normalization'])
    image_transform = NumpyCropToTensor(transform=transform_test, channel_mask=data_config['channel_mask'], **data_config['dataset_kwargs'])
    # 'test': NumpyCropsDataset(data_dir, metadata_test, transform=transform_test, class_names=data_config['class_names'], , **data_config['dataset_kwargs'])

    return image_transform

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    checkpoint_path = 'D:/UW/projects/008_komorki_bialaczka/04_klasyfikacja/wersja_08_PL/results/icm_001/results/size32__default/version_0/checkpoints/last.ckpt'
    config_path = 'D:/UW/projects/008_komorki_bialaczka/04_klasyfikacja/wersja_08_PL/configs/config__size32_shear__default.json'

    config = load_json(config_path)
    model = get_model(checkpoint_path, config).eval()

    image_transform = get_image_transform(config)
