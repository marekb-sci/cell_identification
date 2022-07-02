# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils import tensorboard
import datetime
from pathlib import Path
import sys, os
from copy import deepcopy
import torchmetrics
import pytorch_lightning as pl


from data import NumpyCropsDM, AUGMENTATIONS
import training_utils
from utils import DummyMetric, log_confusion_matrix
from serialize import load_json, save_json

default_config = {
    'data': {
        'data_dir': None,  #set from "paths" file
        'metadata_file': None,  #set from "paths" file
        'data_spit': None, #set from "paths" file D:/UW/projects/008_komorki_bialaczka/03_dane/03_UW_MLset1/04_cropped/data_split_0.json
        'normalization_file': None, #set from "paths" file,
        'normalization': None, #if not None, override settings from normalization_file, e.g. [('Normalize', {'mean': 0, 'std':1000})], # []
        'transform_train': AUGMENTATIONS['flips']+[('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (1, 2), 'shear': 0})]+AUGMENTATIONS['crop32'],
        'transform_test': AUGMENTATIONS['crop32'],
        'channels': {
            'mode': 'interval', # 'mask', 'interval', 'all' ('all' == use all channels)
            'upper_value': 1200, #'inf' for no upper value
            'lower_value': 800 #'-inf' for no lower value
            },
        'subsample_pixels': { #use image at lower resolution (1/2)
            'train': False,
            'test': False
            },
        'class_names': ['Jurkat', 'RPMI8226'], #['Jurkat', 'RPMI8226'], ['B', 'T']
        'dataset_kwargs' : {'img_shape': (48, 48)},
        'train_indices': None,
        'test_indices': None
        },
    'model': {
        'name': 'resnet18',
        'num_classes': 2,
        'in_chans': 1011,
        'pretrained_timm': True,
        'timm_kwargs': {},
        'stem': {
            'timm': 'deep',
            'extra': 'dense',
            'extra_kwargs': {'intermediate_chans': [256, 128]}
            },
        'pretrained_own': False, #True to use own weights (overrides 'pretrained_timm' ),
        'weights_path': None  #set from "paths" file
        },
    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataloader_workers': 4,
        'batch_size': 32,
        'num_epochs': 400,
        'trainer_kwargs': {
            'accumulate_grad_batches': 1,
            'gpus': 1, 'auto_select_gpus': True,
            'gradient_clip_val': 0.5, #'gradient_clip_algorithm': 'value', #https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#gradient-clipping
            'precision': 16
            },
        # 'max_epoch_length': None, #None for use all
        'optimizer': {
            'type': 'AdamW',
            'kwargs': {'lr': 0.001}
            },
        'lr_scheduler': {
            'type': 'StepLR', #OneCycleLR
            'kwargs': {'step_size': 3, 'gamma': 0.97},
            'warm-up_epochs': 100
            },
        'criterion': {
            'type': 'cross_entropy',
            # 'kwargs': {'label_smoothing': 0.1} #'label_smoothing' requires at least pytorch 1.10
            },
        'output': {
            'logging_step': 200,
            'output_dir': None, #set from "paths" file
            'weights_path': None,
            'class_names': None #can be propagated from config['data'] in data prepration
            }
        },
    'output_dir': None #set from "paths" file
    }


#%%
def prepare_config(paths, config=default_config, run_label=None):

    config = deepcopy(config)
    if 'channels' in config['data']:
        prepare_channels(config, paths.get('channels_file'), paths.get('channels_mask_file'))

    #set run_label
    if run_label is None:
        run_label = f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    config['run_label'] = run_label

    #apply paths to config
    output_dir = os.path.join(paths['output_dir'], run_label)
    config['output_dir'] = output_dir
    config['training']['output']['output_dir'] = output_dir

    config['data']['data_dir'] = paths['data_dir']
    config['data']['metadata_file'] = paths['metadata_file']
    config['data']['normalization_file'] = paths.get('normalization_file')
    config['data']['data_split_file'] = paths.get('data_split_file')

    if config['model']['pretrained_own']:
        config['model']['weights_path'] = paths['weights_path']

    # set normalization (settings from file)
    if config['data']['normalization'] is None:
        normalization_settings = load_json(config['data']['normalization_file'])
        config['data']['normalization'] = [('Normalize', normalization_settings)]
    if config['data']['data_split_file'] is not None:
        data_split = load_json(config['data']['data_split_file'])
        config['data']['train_indices'] = data_split['train_indices']
        config['data']['test_indices'] = data_split['test_indices']

    if config['data']['channel_mask'] is not None: # ajust normalizaion to reduced number of channels
        channel_mask = np.array(config['data']['channel_mask'])
        mean = np.array(config['data']['normalization'][0][1]['mean'])
        std = np.array(config['data']['normalization'][0][1]['std'])
        if len(mean) == len(channel_mask):
            config['data']['normalization'][0][1]['mean'] = mean[channel_mask].tolist()
        if len(std) == len(channel_mask):
            config['data']['normalization'][0][1]['std'] = std[channel_mask].tolist()

    # propagate values in config
    config['training']['output']['class_names'] = config['data']['class_names']

    # system-specific setting
    if sys.platform == 'win32':
        config['training']['dataloader_workers'] = 0

    return config

def prepare_channels(config, channels_file, channels_mask_file): #modifiy config in place

    if config['data']['channels']['mode'] == 'all':
        channel_mask = None

    elif config['data']['channels']['mode'] == 'mask':
        channel_mask = np.load(channels_mask_file).tolist()

    elif config['data']['channels']['mode'] == 'interval':
        channels_values = np.loadtxt(channels_file)
        channel_mask = ((channels_values >= float(config['data']['channels']['lower_value'])) & (channels_values <= float(config['data']['channels']['upper_value']))).tolist()
    else:
        raise NotImplementedError()

    # apply channel mask and number of input channels
    config['data']['channel_mask'] = channel_mask
    if channel_mask is not None:
        config['model']['in_chans'] = sum(channel_mask)


#%%

class ParticlesClassifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()


        self.model = training_utils.get_model(config['model'])

        self.optimizer_config = config['training']['optimizer']
        self.lr_scheduler_config = config['training']['lr_scheduler']

        self.criterion = training_utils.get_criterion(config['training']['criterion'])

        self.setup_metrics(num_classes=config['model']['num_classes'])
        self.class_names = config['training']['output'].get('class_names')


    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self.model(x)
        loss = self.criterion(p, y)
        self.update_metrics(loss, y, p)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p = self.model(x)
        loss = self.criterion(p, y)
        self.update_metrics(loss, y, p)

        return {'loss': loss}


    def setup_metrics(self, num_classes):
            self.metrics = {
                'accuracy': torchmetrics.Accuracy(),
                'loss': DummyMetric(),
                'confusion matrix': torchmetrics.ConfusionMatrix(
                    num_classes=num_classes, normalize='none' #'true'
                    )
                }

    @torch.no_grad()
    def update_metrics(self, loss, y, y_hat):
        loss, y, y_hat = loss.cpu().float(), y.cpu() , y_hat.cpu().float()
        self.metrics['loss'](loss, len(y))
        self.metrics['accuracy'](y_hat.softmax(dim=-1), y)
        self.metrics['confusion matrix'](y_hat.softmax(dim=-1), y)


    def log_metrics(self, prefix='', reset=False):
        metrics = {}
        for label, metric in self.metrics.items():
             #must be logged in different way
            metric_value = metric.compute()

            if label == 'confusion matrix':
                log_confusion_matrix(
                    metric_value.cpu().numpy(),
                    self.logger.experiment,
                    class_names = self.class_names,
                    num_classes = metric.num_classes,
                    image_label = f'{prefix}confusion matrix',
                    epoch = self.current_epoch) #self.global_step)
            else:
                self.log(f'{prefix}{label}', metric_value)
                metrics[f'{prefix}{label}'] = metric.compute()
            if reset:
                metric.reset()

        return metrics

    def training_epoch_end(self, training_step_outputs):
        self.log_metrics(prefix='train_', reset=True)

    def validation_epoch_end(self, validation_step_outputs):
        self.log_metrics(prefix='val_', reset=True)
        # self.logger.log_hyperparams(self.hparams, metrics)

    def configure_optimizers(self):
        # for multi stage training: https://forums.pytorchlightning.ai/t/what-is-the-best-way-to-train-on-stages/95/2

        optimizer, scheduler = training_utils.get_optimizer_and_scheduler(
            self.model,
            self.optimizer_config,
            self.lr_scheduler_config
            )

        return [optimizer], [scheduler]

    def load_weights_from_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt['state_dict'])



#%%
if __name__=='__main__':

    default_paths = load_json('default_paths.json')
    my_config = prepare_config(default_paths, config=default_config)

    #uncomment to test step by step
    # fold_configs = create_folds_configs(my_config)
    # datasets = get_datasets(fold_configs[0]['data'])
    # model = training_utils.get_timm_model(fold_configs[0]['model'])

    data_module = NumpyCropsDM(my_config['data'])
    model = ParticlesClassifier(my_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=3, save_last=True)
    logger_dir, logger_name = os.path.split(my_config['output_dir'])

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        # default_root_dir = Config.log_dir,
        logger = pl.loggers.TensorBoardLogger(logger_dir, name=logger_name, default_hp_metric=False),
        max_epochs = my_config['training']['num_epochs'],
        **my_config['training']['trainer_kwargs']
        )
    # trainer = Trainer(default_root_dir='/your/path/to/save/checkpoints')

    save_json(my_config, Path(trainer.log_dir) / 'config.json')
    trainer.fit(model, datamodule=data_module)



    # training_output = run_kfold_training(my_config)
    # save_json(training_output, os.path.join(my_config['output_dir'], 'training_output.json'))



