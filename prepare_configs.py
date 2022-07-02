# -*- coding: utf-8 -*-
import os
from pathlib import Path
from copy import deepcopy
from itertools import product

from serialize import load_json, save_json
from data import AUGMENTATIONS
from training import default_config, prepare_config


augmentation_options = {
    'size128_shear': { #large (128)
         'dataset_kwargs': {'img_shape': (48, 48)},
         'transform_train': AUGMENTATIONS['flips']+ \
             [('Resize', {'size': 144}),
              ('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (0.8, 1.2), 'shear': 0.1}),
              ('CenterCrop', {'size': 128})],
         'transform_test': [('Resize', {'size': 144}), ('CenterCrop', {'size': 128})]
     },
    'size128':{ # no shear, 128
         'dataset_kwargs': {'img_shape': (48, 48)},
         'transform_train': AUGMENTATIONS['flips']+ \
             [('Resize', {'size': 144}),
              ('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (0.8, 1.2), 'shear': 0.0}),
              ('CenterCrop', {'size': 128})],
         'transform_test': [('Resize', {'size': 144}), ('CenterCrop', {'size': 128})]
     },
    'size32':{ # no shear, 32
         'dataset_kwargs': {'img_shape': (48, 48)},
         'transform_train': AUGMENTATIONS['flips']+ \
             [('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (0.8, 1.2), 'shear': 0.0}),
              ('CenterCrop', {'size': 32})],
         'transform_test': [('CenterCrop', {'size': 32})]
     },
    'size32_shear':{
        'dataset_kwargs': {'img_shape': (48, 48)},
         'transform_train': AUGMENTATIONS['flips']+ \
             [('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (0.8, 1.2), 'shear': 0.1}),
              ('CenterCrop', {'size': 32})],
         'transform_test': [('CenterCrop', {'size': 32})]
        }
}

model_options = {
    'default': {
        'name': 'resnet18',
        'stem': {
            'timm': '',
            'extra': 'add_linear',
            'extra_kwargs': {}
            }
        },
    'timm': {
        'name': 'resnet18',
        'stem': {
            'timm': '',
            'extra': '',
            'extra_kwargs': {}
            }
        },
    'timm_deep': {
        'name': 'resnet18',
        'stem': {
            'timm': 'deep',
            'extra': '',
            'extra_kwargs': {}
            }
        },
    'dense': {
        'name': 'resnet18',
        'stem': {
            'timm': '',
            'extra': 'dense',
            'extra_kwargs': {'intermediate_chans': [256, 128]}
            }
        },
    'dense_dropout': {
        'name': 'resnet18',
        'stem': {
            'timm': '',
            'extra': 'dense',
            'extra_kwargs': {'intermediate_chans': [256, 128], 'dropout': 0.5}
            }
        },
    'dense_deep': {
        'name': 'resnet18',
        'stem': {
            'timm': 'deep',
            'extra': 'dense',
            'extra_kwargs': {'intermediate_chans': [256, 128]}
            }
        },
    'densenet_deep': {
        'name': 'densenet121',
        'stem': {
            'timm': 'deep',
            'extra': '',
            'extra_kwargs': {}
            }
        },
}

PATHS = load_json('paths_icm.json')
OUTPUT_DIR = Path('configs')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

base_config = default_config
base_config['data']['channels'] = {'mode': 'all'}
base_config['training']['dataloader_workers'] = 16 
base_config['training']['batch_size'] = 64

for augmentation_label, model_label in product(augmentation_options, model_options):
    run_label = f'{augmentation_label}__{model_label}'
    config = prepare_config(PATHS, config=deepcopy(base_config), run_label=run_label)

    config['data'].update(augmentation_options[augmentation_label])
    config['model'].update(model_options[model_label])

    save_json(config, OUTPUT_DIR / f'config__{run_label}.json')
