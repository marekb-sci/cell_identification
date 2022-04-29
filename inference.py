# -*- coding: utf-8 -*-
from pathlib import Path
import torch
import numpy as np
from scipy.special import softmax

import serialize
from training_utils import get_timm_model
from data import NumpyCropToTensor, get_augmentation_tv

def get_image_transform(data_config):
    transform_test = get_augmentation_tv(data_config['transform_test']+data_config['normalization'])
    image_transform = NumpyCropToTensor(transform=transform_test, channel_mask=data_config['channel_mask'], **data_config['dataset_kwargs'])
    return image_transform

# this example works on cpu
if __name__== '__main__':
    torch.set_grad_enabled(False) #disable gradient computation (faster inference, less memory used)

    # path to the run config (not "kfold_config") (e.g. '.../run_.../run_..._0_config.json')
    config_path = ''
    config = serialize.load(config_path)

    # path to model weights (e.g. '.../run_.../run_..._0_last.pt')
    weights_paths = '' 

    # path to directory with images to classification (e.g. .../04_cropped/images)
    images_dir = ''

    # get model
    model_config = config['model']
    model_config['pretrained_own'] = True
    model_config['weights_path'] = weights_paths
    model = get_timm_model(config['model']).eval()

    image_transform = get_image_transform(config['data'])
    class_names = config['data']['class_names']

    # images_all = list(Path(r'D:\UW\projects\008_komorki_bialaczka\03_dane\03_UW_MLset1\04_cropped\images').iterdir())
    images_all = list(Path(images_dir).iterdir())
    results = []
    for fn in images_all:
        img = image_transform(fn)
        out = model(img.unsqueeze(0))[0].cpu().numpy()
        results.append(out)
        class_idx = np.argmax(out)
        print(f'{fn.name}, pred: {class_names[class_idx]}, score: {softmax(out).max():0.4f}')
