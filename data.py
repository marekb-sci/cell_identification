# -*- coding: utf-8 -*-
import numpy as np
import torch.utils.data as data
import os
import torch
import pandas as pd
from pathlib import Path
import torchvision

def get_augmentation_tv(aug_list):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.__getattribute__(aug[0])(**aug[1]) for aug in aug_list
        ])
    return transform

AUGMENTATIONS = {
    'flips': [('RandomHorizontalFlip', {}), ('RandomVerticalFlip', {})],
    'rotation': [('RandomRotation', {'degrees': 45})],
    'affine': [('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (1, 2), 'shear': 0})],
    'crop32':  [('CenterCrop', {'size': 32})]
}


def min_bcg_generator(img_shape, img_in):
    img_out = np.zeros(img_shape)
    img_out[:] = img_in.min(axis=(1,2), keepdims=True)
    return np.full(img_shape, np.median(img_in))

def zero_bcg_generator(img_shape, img_in):
    return np.zeros(img_shape)


class NumpyCropsDataset(torch.utils.data.Dataset):
    """
    metadata: file_name, label
    """
    BCG_GENERATORS = {
        None: None,
        'zero': zero_bcg_generator,
        'min': min_bcg_generator,
    }
    def __init__(self, data_dir, metadata, img_shape=(1024, 48, 48),
                 swap_input_axis=True, #for h,w,c input (h,w,c -> c,h,w)
                 transform=None,
                 target_transform=None,
                 indices=None,
                 background_generator='zero',
                 class_names = ['B', 'T']
                 ):

        self.data_dir = Path(data_dir)

        if isinstance(metadata, pd.DataFrame):
            self.metadata = metadata
        else:
            self.metadata = pd.read_csv(metadata)

        self.img_shape = img_shape

        self.class_name_to_number = {name: i for i, name in enumerate(class_names)}

        self.swap_input_axis = swap_input_axis
        self.transform = transform
        self.target_transform = target_transform

        if indices is None:
            indices = np.arange(len(self.metadata))
        self.indices = indices

        if background_generator in self.BCG_GENERATORS:
            self.background_generator = self.BCG_GENERATORS[background_generator]
        else:
            assert callable(background_generator)
            self.background_generator = background_generator

    def __getitem__(self, idx_raw):
        idx = self.indices[idx_raw]
        metadata = self.metadata.iloc[idx]
        fn = metadata['file_name']

        #load image
        img_in = np.load(self.data_dir / fn)
        if self.swap_input_axis:
            img_in = np.moveaxis(img_in, 2, 0)

        #generate background and paste input file inside
        if self.background_generator is not None:
            img = self.background_generator(self.img_shape, img_in)
            assert (img.shape[1] >= img_in.shape[1]) and (img.shape[2] >= img_in.shape[2])
            margin_top = (img.shape[1] - img_in.shape[1])//2
            margin_left = (img.shape[2] - img_in.shape[2])//2
            img[:,margin_top:margin_top+img_in.shape[1], margin_left:margin_left+img_in.shape[2]] = img_in
        else:
            img = img_in


        tensor = torch.tensor(img, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if self.transform is not None:
            tensor = self.transform(tensor)

        target = self.class_name_to_number[metadata.get('label')]


        return tensor, target

    def __len__(self):
        return len(self.indices)