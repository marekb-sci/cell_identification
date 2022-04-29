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

from data import NumpyCropsDataset, NumpyCropToTensor, AUGMENTATIONS, get_augmentation_tv, subsample_pixels
import training_utils
from utils import DummyMetric, log_confusion_matrix
from serialize import load_json, save_json

# add crop 96
AUGMENTATIONS['crop96'] = [('CenterCrop', {'size': 96})]
# modify affine (smaller resize)
# AUGMENTATIONS['affine'] = [('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (0.8, 1.2), 'shear': 0})]


default_config = {
    'data': {
        'data_dir': None,  #set from "paths" file
        'metadata_file': None,  #set from "paths" file
        'k_folds': 5,
        'normalization_file': None, #set from "paths" file,
        'normalization': None, #if not None, override settings from normalization_file, e.g. [('Normalize', {'mean': 0, 'std':1000})], # []
        'transform_train': AUGMENTATIONS['flips']+[('RandomAffine', {'degrees': 45, 'translate': (0.1, 0.1), 'scale': (1, 2), 'shear': 0})]+AUGMENTATIONS['crop32'],
        'transform_test': AUGMENTATIONS['crop32'],
        'channels': {
            'mode': 'interval', # 'mask', 'interval', 'all' ('all' == use all channels)
            'upper_value': 1200, #'inf' for no upper value
            'lower_value': 800, #'-inf' for no lower value
            'mask': None # list [True, False, ...] with True/False for each channel, where True indicate which channels will be used. Override mask from channel_mask_file
            },
        'subsample_pixels': { #use image at lower resolution (1/2)
            'train': False,
            'test': False
            },
        'class_names': ['Jurkat', 'RPMI8226'], #['Jurkat', 'RPMI8226'], ['B', 'T']
        'dataset_kwargs' : {'img_shape': (48, 48)}
        },
    'model': {
        'name': 'resnet18',
        'num_classes': 2,
        'in_chans': 1011,
        'pretrained_timm': True,
        'timm_kwargs': {},
        'linear_channels_adapter': True, #add initial linear layer mapping channels to RGB
        'pretrained_own': False, #True to use own weights (overrides 'pretrained_timm' ),
        'weights_path': None  #set from "paths" file
        },
    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataloader_workers': 4,
        'batch_size': 32,
        'num_epochs': 400,
        'max_epoch_length': None, #None for use all
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

    if config['model']['pretrained_own']:
        config['model']['weights_path'] = paths['weights_path']

    # set normalization (settings from file)
    if config['data']['normalization'] is None:
        normalization_settings = load_json(config['data']['normalization_file'])
        config['data']['normalization'] = [('Normalize', normalization_settings)]
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
        if config['data']['channels'] is not None:
            channel_mask = config['data']['channels']
        else:
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

def generate_data_splits(metadata, n_splits, one_image_one_dataset=False):
    """one_image_one_dataset: put data cropped from the same image to the same dataset"""
    if one_image_one_dataset:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.arange(len(metadata)), metadata['label'].to_numpy())

    parent_image_labels = metadata.groupby('parent_image').first()['label']
    indices_all = np.arange(len(metadata))

    splits = []
    for img_train_indices, img_test_indices in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(parent_image_labels.index.to_numpy(), parent_image_labels.to_numpy()):
        test_parent_imgs = parent_image_labels.index[img_test_indices]

        test_mask = metadata.parent_image.isin(test_parent_imgs)
        train_indices = indices_all[~test_mask]
        test_indices = indices_all[test_mask]
        splits.append([train_indices, test_indices])

    return splits


def create_folds_configs(config):
    metadata = pd.read_csv(config['data']['metadata_file'])

    data_split = generate_data_splits(metadata, config['data']['k_folds']) #StratifiedKFold(n_splits=config['data']['k_folds'], shuffle=True, random_state=0).split(np.arange(len(metadata)), metadata['label'].to_numpy())

    folds_configs = []
    for i_fold, (train_indices, test_indices) in enumerate(data_split):
        fold_config = deepcopy(config)

        fold_config['data']['train_indices'] = train_indices.tolist()
        fold_config['data']['test_indices'] = test_indices.tolist()

        fold_config['training']['suffix'] = f' {i_fold}' #this will be added to logged values (e.g. 'loss 2') and weight file names

        folds_configs.append(fold_config)

    return folds_configs

def get_datasets(data_config):
    metadata = pd.read_csv(data_config['metadata_file'])

    transform_train = get_augmentation_tv(data_config['transform_train']+data_config['normalization'])
    transform_test = get_augmentation_tv(data_config['transform_test']+data_config['normalization'])

    data_dir = data_config['data_dir']

    metadata_train = metadata.iloc[data_config['train_indices']]
    metadata_test = metadata.iloc[data_config['test_indices']]

    datasets = {
        'train': NumpyCropsDataset(data_dir, metadata_train, transform=transform_train, class_names=data_config['class_names'], channel_mask=data_config['channel_mask'], **data_config['dataset_kwargs']),
        'test': NumpyCropsDataset(data_dir, metadata_test, transform=transform_test, class_names=data_config['class_names'], channel_mask=data_config['channel_mask'], **data_config['dataset_kwargs'])
        }

    for ds_type in ['train', 'test']:
        if 'subsample_pixels' in data_config and data_config['subsample_pixels'][ds_type]:
            datasets[ds_type] = subsample_pixels(datasets[ds_type])

    return datasets

#%%
def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, train_config, tb_logger, metrics, state):
    device = train_config['device']
    model.train()
    trainng_steps_beginning = state['training_steps']
    suffix = state.get('suffix', '')

    tb_logger.add_scalars('lr', {f'lr{suffix}': optimizer.param_groups[0]['lr']}, state['training_steps'])
    for x, target in data_loader:
        target = target.to(device)
        y = model(x.to(device))
        loss = criterion(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state['training_steps'] += len(target)
        metrics['loss'](loss, len(target))
        metrics['accuracy'](y.softmax(dim=-1), target)
        metrics['confusion matrix'](y.softmax(dim=-1), target)

        if state['training_steps'] - state['last_log'] >= train_config['output']['logging_step']:
            state['last_log'] = state['training_steps']

            #logging
            tb_logger.add_scalars('loss', {f'train{suffix}': metrics['loss'].compute()}, state['training_steps'])
            tb_logger.add_scalars('accuracy', {f'train{suffix}': metrics['accuracy'].compute()}, state['training_steps'])
            log_confusion_matrix(metrics['confusion matrix'].compute().cpu().numpy(),
                     tb_logger, class_names=train_config['output'].get('class_names'),
                     num_classes = metrics['confusion matrix'].num_classes,
                     image_label=f'train confusion matrix{suffix}', epoch=state['training_steps'])

            tb_logger.flush()

            metrics['loss'].reset()
            metrics['accuracy'].reset()
            metrics['confusion matrix'].reset()

        if train_config['max_epoch_length'] is not None and state['training_steps'] - trainng_steps_beginning >= train_config['max_epoch_length']:
            break

    if scheduler is not None:
        scheduler.step()

    return state


#%%

@torch.no_grad()
def validate(model, data_loader, criterion, train_config, tb_logger, metrics, state):
    device = train_config['device']
    model.eval()
    for x, target in data_loader:
        target = target.to(device)
        y = model(x.to(device))
        loss = criterion(y, target)

        metrics['loss'](loss, len(x))
        metrics['accuracy'](y.softmax(dim=-1), target)
        metrics['confusion matrix'](y.softmax(dim=-1), target)

    #logging
    val_metrics = {'loss': metrics['loss'].compute(),  'accuracy': metrics['accuracy'].compute(),
                   'cm': metrics['confusion matrix'].compute().cpu().numpy()}

    suffix = state.get('suffix', '')
    tb_logger.add_scalars('loss',  {f'val{suffix}': val_metrics['loss']}, state['training_steps'])
    tb_logger.add_scalars('accuracy', {f'val{suffix}': val_metrics['accuracy']}, state['training_steps'])

    class_names = train_config['output'].get('class_names')
    log_confusion_matrix(val_metrics['cm'],
                         tb_logger, class_names=class_names,
                         num_classes = metrics['confusion matrix'].num_classes,
                         image_label=f'val confusion matrix{suffix}', epoch=state['epoch'])

    tb_logger.flush()

    metrics['accuracy'].reset()
    metrics['loss'].reset()
    metrics['confusion matrix'].reset()

    return val_metrics

#%%

def run_kfold_training(config):
    output_dir = Path(config['training']['output']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    save_json(config, output_dir / f'{config["run_label"]}_kfold_config.json')

    folds_configs = create_folds_configs(config)

    results = []
    for fold_config in folds_configs:
        fold_result = run_training(fold_config)
        results.append(fold_result)

    test_acc_sum = 0
    test_acc_weight = 0
    for fold_result, fold_config in zip(results, folds_configs):
        weight = len(fold_config['data']['test_indices'])
        test_acc_sum += weight * fold_result[-1]['val_accuracy']
        test_acc_weight += weight
    test_acc = test_acc_sum / test_acc_weight
    print(f'overal test accuracy: {test_acc:0.4f}')

    return results

def run_training(config):

    output_dir = Path(config['training']['output']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    config_file_name = f'{config["run_label"]}{config["training"]["suffix"].replace(" ","_")}_config.json'
    save_json(config, output_dir / config_file_name)

    datasets = get_datasets(config['data'])
    ds_train = datasets['train']
    ds_val = datasets['test']

    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True,
                                           batch_size=config['training']['batch_size'],
                                           pin_memory=True,
                                           num_workers=config['training']['dataloader_workers'],
                                           drop_last=True #for timm with 32x32 or smaller input, error occures when batch of size 1 is given in training mode
                                           )

    dl_val = torch.utils.data.DataLoader(ds_val, shuffle=False,
                                       batch_size=config['training']['batch_size'],
                                       pin_memory=True,
                                       num_workers=config['training']['dataloader_workers']
                                       )

    model = training_utils.get_timm_model(config['model'])

    optimizer, scheduler = training_utils.get_optimizer_and_scheduler(model,
                                                       config['training']['optimizer'],
                                                       config['training']['lr_scheduler'])
    criterion = training_utils.get_criterion(config['training']['criterion'])

    train_metrics = {
        'accuracy': torchmetrics.Accuracy().to(config['training']['device']),
        'loss': DummyMetric().to(config['training']['device']),
        'confusion matrix': torchmetrics.ConfusionMatrix(
            num_classes=config['model']['num_classes'], normalize='none' #'true'
            ).to(config['training']['device'])
        }

    val_metrics = {
        'accuracy': torchmetrics.Accuracy().to(config['training']['device']),
        'loss': DummyMetric().to(config['training']['device']),
        'confusion matrix': torchmetrics.ConfusionMatrix(
            num_classes=config['model']['num_classes'], normalize='none' #'true'
            ).to(config['training']['device'])
        }

    tb_logger = tensorboard.SummaryWriter(config['training']['output']['output_dir'])
    checkpoint_saver = training_utils.CheckpointSaver(
        lambda score: os.path.join(config['training']['output']['output_dir'], f'{config["run_label"]}{config["training"]["suffix"].replace(" ","_")}_best_{score:0.2f}.pt'),
        config['training']['device']
        )

    model = model.to(config['training']['device'])
    state = {'training_steps': 0, 'epoch': None, 'last_log': 0, 'suffix': config['training']['suffix']}
    metrics_history = []
    # best_metrics = None
    time_start = datetime.datetime.now()
    #TRAINING LOOP
    for i_epoch in range(config['training']['num_epochs']):
        state['epoch'] = i_epoch
        state = train_one_epoch(model, dl_train, criterion, optimizer, scheduler, config['training'], tb_logger, train_metrics, state)
        val_metrics_values = validate(model, dl_val, criterion, config['training'], tb_logger, val_metrics, state)

        # save weights if best so far
        is_best = checkpoint_saver.save_if_best(model, val_metrics_values['accuracy'])

        #prepare metrics for logging
        metrics_out = {'val_accuracy': val_metrics_values['accuracy'].cpu().item(),
                       'val_loss': val_metrics_values['loss'].cpu().item(),
                       'time': (datetime.datetime.now() - time_start).total_seconds(),
                       'num_epoch': i_epoch
                       }
        # #test evaluation for best
        # if is_best:
        #     test_metrics = validate(model, dl_test, criterion, config['training'], tb_logger, metrics, state)
        #     metrics_out['test_accuracy'] = test_metrics['accuracy'].cpu().item()
        #     metrics_out['test_loss'] = test_metrics['loss'].cpu().item()
        #     best_metrics = metrics_out

        metrics_history.append(metrics_out)
    #END OF TRAINING LOOP
    weights_file_name = f'{config["run_label"]}{config["training"]["suffix"].replace(" ","_")}_last_{metrics_out["val_accuracy"]:0.2f}.pt'
    torch.save(model.cpu().state_dict(), output_dir / weights_file_name)

    # tb_logger.add_hparams(get_hparams(config), best_metrics)
    tb_logger.close()

    #save metrics to json
    metrics_file_name = f'{config["run_label"]}{config["training"]["suffix"].replace(" ","_")}_results.json'
    save_json(metrics_history, output_dir / metrics_file_name)

    return metrics_history

#%%
if __name__=='__main__':
    paths = load_json('paths_test.json')
    my_config = prepare_config(paths, config=default_config)

    #uncomment to test step by step
    # fold_configs = create_folds_configs(my_config)
    # datasets = get_datasets(fold_configs[0]['data'])
    # model = training_utils.get_timm_model(fold_configs[0]['model'])

    training_output = run_kfold_training(my_config)
    save_json(training_output, os.path.join(my_config['output_dir'], 'training_output.json'))



