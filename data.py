# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import torch.utils.data as data
import os
from copy import deepcopy
import torch
import pandas as pd
from pathlib import Path
import torchvision
import pytorch_lightning as pl


def get_augmentation_tv(aug_list):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.__getattribute__(aug[0])(**aug[1]) for aug in aug_list
        ])
    return transform

AUGMENTATIONS = {
    'flips': [('RandomHorizontalFlip', {}), ('RandomVerticalFlip', {})],
    'rotation': [('RandomRotation', {'degrees': 45})],
    'affine': [('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (1, 2), 'shear': 0})],
    'crop32': [('CenterCrop', {'size': 32})],
    'crop96': [('CenterCrop', {'size': 96})],
}


def min_bcg_generator(img_shape, img_in):
    num_chans = img_in.shape[0]
    img_out = np.zeros((num_chans, *img_shape))
    img_out[:] = img_in.min(axis=(1,2), keepdims=True)
    return np.full(img_shape, np.median(img_in))

def zero_bcg_generator(img_shape, img_in):
    num_chans = img_in.shape[0]
    return np.zeros((num_chans, *img_shape))

class NumpyCropToTensor:
    BCG_GENERATORS = {
        None: None,
        'zero': zero_bcg_generator,
        'min': min_bcg_generator,
        }

    def __init__(self,
                 img_shape=(48, 48),
                 swap_input_axis=True, #for h,w,c input (h,w,c -> c,h,w)
                 transform=None,
                 channel_mask=None,
                 background_generator='zero'
                 ):

        self.img_shape = img_shape

        self.swap_input_axis = swap_input_axis
        self.transform = transform

        if channel_mask is None:
            channel_mask = ... #'...' means 'take all channels'
        self.channel_mask = channel_mask

        if background_generator in self.BCG_GENERATORS:
            self.background_generator = self.BCG_GENERATORS[background_generator]
        else:
            assert callable(background_generator)
            self.background_generator = background_generator

    def __call__(self, file_name):
        """prepare tensor from numpy file """
        img_in = np.load(file_name)

        if self.swap_input_axis:
            img_in = np.moveaxis(img_in, 2, 0)
        img_in = img_in[self.channel_mask]

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

        return tensor



class NumpyCropsDataset(torch.utils.data.Dataset):
    """
    metadata: file_name, label
    """

    def __init__(self, data_dir, metadata,
                 img_shape=(48, 48),
                 swap_input_axis=True, #for h,w,c input (h,w,c -> c,h,w)
                 transform=None,
                 target_transform=None,
                 channel_mask=None,
                 indices=None,
                 background_generator='zero',
                 class_names = ['B', 'T']
                 ):


        self.data_dir = Path(data_dir)

        if isinstance(metadata, pd.DataFrame):
            self.metadata = metadata
        else:
            self.metadata = pd.read_csv(metadata)

        self.numpy_to_tensor = NumpyCropToTensor(
            img_shape, swap_input_axis, transform, channel_mask, background_generator
            )
        self.class_name_to_number = {name: i for i, name in enumerate(class_names)}
        self.target_transform = target_transform

        if indices is None:
            indices = np.arange(len(self.metadata))
        self.indices = indices

    def __getitem__(self, idx_raw):
        idx = self.indices[idx_raw]
        metadata = self.metadata.iloc[idx]
        fn = metadata['file_name']

        #prepare image
        tensor = self.numpy_to_tensor(self.data_dir / fn)
        #prepare target
        target = self.class_name_to_number[metadata.get('label')]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return tensor, target

    def __len__(self):
        return len(self.indices)

def subsample_pixels(ds):
    datasets = []
    for i in [0,1]:
        for j in [0,1]:
            ds_new = deepcopy(ds)
            ds_new.numpy_transform = lambda img: np.kron(img[i::2,j::2], np.ones((2,2,1))) #partial(subsample_and_expand, i=i, j=j)
            datasets.append(ds_new)

    ds_out = torch.utils.data.ConcatDataset(datasets=datasets)
    ds_out.metadata = ds.metadata
    return ds_out

class NumpyCropsDM(pl.LightningDataModule):
    def __init__(self, data_config, batch_size=32, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup_data(data_config)

    def setup_data(self, data_config):

        self.data_config = data_config

        metadata = pd.read_csv(data_config['metadata_file'])
        data_dir = data_config['data_dir']

        self.transform_train = get_augmentation_tv(data_config['transform_train']+data_config['normalization'])
        self.transform_test = get_augmentation_tv(data_config['transform_test']+data_config['normalization'])

        self.metadata_train = metadata.iloc[data_config['train_indices']]
        self.metadata_test = metadata.iloc[data_config['test_indices']]

        self.class_names = data_config['class_names']
        self.channel_mask = data_config['channel_mask']

        self.dataset_train = NumpyCropsDataset(data_dir, self.metadata_train, transform=self.transform_train, class_names=self.class_names, channel_mask=self.channel_mask, **data_config['dataset_kwargs'])
        self.dataset_val = NumpyCropsDataset(data_dir, self.metadata_test, transform=self.transform_test, class_names=self.class_names, channel_mask=self.channel_mask, **data_config['dataset_kwargs'])

        subsample_pixels_config = data_config.get('subsample_pixels', {'train': False, 'test': False})
        if subsample_pixels_config['train']:
            self.dataset_train = subsample_pixels(self.dataset_train)
        if subsample_pixels_config['test']:
            self.dataset_val = subsample_pixels(self.dataset_val)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_train,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_val,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers
                                           )
