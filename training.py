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

from data import NumpyCropsDataset, AUGMENTATIONS, get_augmentation_tv
import training_utils
from utils import DummyMetric, log_confusion_matrix
from serialize import load_json, save_json


PATHS = load_json('paths.json')

config = {
    'data': {
        'data_dir': PATHS['data_dir'],
        'metadata_file': PATHS['metadata_file'],
        'k_folds': 5,
        'normalization': [('Normalize', {'mean': 0, 'std':1000})], # []
        'transform_train': AUGMENTATIONS['flips']+AUGMENTATIONS['affine']+AUGMENTATIONS['crop32'],
        'transform_test': AUGMENTATIONS['crop32'],
        'channels': {
            'channels_file': PATHS['channels_file'],
            'upper_value': 1200, #'inf' for no upper value
            'lower_value': 800 #'-inf' for no lower value
            }
        },
    'model': {
        'name': 'resnet18',
        'num_classes': 2,
        'in_chans': 1024,
        'pretrained': True,
        'initial_dropout': 0.0,
        'reduce_channels': True,
        'timm_kwargs': {}
        #TODO: add weights path
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
            'type': 'cross_entropy'
            },
        'output': {
            'logging_step': 200,
            'output_dir': None,
            # 'weights_path': None,
            'class_names': ['B', 'T']
            }
        },
    'output_dir': PATHS['output_dir']
    }

#%%
def prepare_config(config=config, run_label=None):

    config = deepcopy(config)

    prepare_channels(config)

    if run_label is None:
        run_label = f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    if sys.platform == 'win32':
        config['training']['dataloader_workers'] = 0
    config['run_label'] = run_label

    config['output_dir'] = os.path.join(config['output_dir'], run_label)
    config['training']['output']['output_dir'] = config['output_dir']

    return config

def prepare_channels(config): #modifiy config in place

    # get channel mask and number of input channels
    if 'channels' in config['data']:
        channels_values = np.loadtxt(config['data']['channels']['channels_file'])
        channel_mask = ((channels_values >= float(config['data']['channels']['lower_value'])) & (channels_values <= float(config['data']['channels']['upper_value']))).tolist()
        in_chans = sum(channel_mask)
    else: #use all channels
        channel_mask = None
        in_chans = 1024

    # apply channel mask and number of input channels
    config['data']['channel_mask'] = channel_mask
    config['model']['in_chans'] = in_chans


#%%
def get_fold_datasets(data_config):
    metadata = pd.read_csv(data_config['metadata_file'])
    parent_image_labels = metadata.groupby('parent_image').first()['label']

    transform_train = get_augmentation_tv(data_config['transform_train']+data_config['normalization'])
    transform_test = get_augmentation_tv(data_config['transform_test']+data_config['normalization'])

    data_dir = data_config['data_dir']
    channel_mask = data_config['channel_mask']

    datasets = []
    for train_indices, test_indices in StratifiedKFold(n_splits=data_config['k_folds'], shuffle=True, random_state=0).split(parent_image_labels.index.to_numpy(), parent_image_labels.to_numpy()):
        test_parent_imgs = parent_image_labels.index[test_indices]

        test_mask = metadata.parent_image.isin(test_parent_imgs)
        metadata_train = metadata.loc[~test_mask]
        metadata_test = metadata.loc[test_mask]

        datasets.append({
            'train': NumpyCropsDataset(data_dir, metadata_train, transform=transform_train, channel_mask=channel_mask),
            'test': NumpyCropsDataset(data_dir, metadata_test, transform=transform_test, channel_mask=channel_mask)
            })

    return datasets

#%%
def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, train_config, tb_logger, metrics, state):
    device = train_config['device']
    model.train()
    trainng_steps_beginning = state['training_steps']
    i_fold = state.get('i_fold', 0)

    tb_logger.add_scalars('lr', {f'fold_{i_fold}': optimizer.param_groups[0]['lr']}, state['training_steps'])
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
            tb_logger.add_scalars('loss', {f'train_{i_fold}': metrics['loss'].compute()}, state['training_steps'])
            tb_logger.add_scalars('accuracy', {f'train_{i_fold}': metrics['accuracy'].compute()}, state['training_steps'])
            log_confusion_matrix(metrics['confusion matrix'].compute().cpu().numpy(),
                     tb_logger, class_names=train_config['output'].get('class_names'),
                     num_classes = metrics['confusion matrix'].num_classes,
                     image_label=f'train confusion matrix, fold {i_fold}', epoch=state['training_steps'])

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

    i_fold = state.get('i_fold', 0)
    tb_logger.add_scalars('loss',  {f'val_{i_fold}': val_metrics['loss']}, state['training_steps'])
    tb_logger.add_scalars('accuracy', {f'val_{i_fold}': val_metrics['accuracy']}, state['training_steps'])

    class_names = train_config['output'].get('class_names')
    log_confusion_matrix(val_metrics['cm'],
                         tb_logger, class_names=class_names,
                         num_classes = metrics['confusion matrix'].num_classes,
                         image_label=f'val confusion matrix, fold {i_fold}', epoch=state['epoch'])

    tb_logger.flush()

    metrics['accuracy'].reset()
    metrics['loss'].reset()
    metrics['confusion matrix'].reset()

    return val_metrics

#%%
def run_training_for_fold(config, dl_train, dl_val, i_fold):

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
        lambda score: os.path.join(config['training']['output']['output_dir'], f'{config["run_label"]}_fold{i_fold}_best_{score:0.2f}.pt'),
        config['training']['device']
        )

    model = model.to(config['training']['device'])
    state = {'training_steps': 0, 'epoch': None, 'last_log': 0, 'i_fold': i_fold}
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

    # tb_logger.add_hparams(get_hparams(config), best_metrics)
    tb_logger.close()

    return metrics_history

#%%
def run_training(config):
    datasets = get_fold_datasets(config['data'])

    output_dir = Path(config['training']['output']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    save_json(config, output_dir / 'config.json')


    results = []
    for i_fold, fold_datasets in enumerate(datasets):
        ds_train = fold_datasets['train']
        ds_val = fold_datasets['test']



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

        fold_history = run_training_for_fold(config, dl_train, dl_val, i_fold)

        data_split = {'train': fold_datasets['train'].metadata.index.to_list(),
                      'test': fold_datasets['test'].metadata.index.to_list()}

        fold_result = {'history': fold_history, 'data_split': data_split}
        results.append(fold_result)

    return results

#%%
if __name__=='__main__':
    my_config = prepare_config()
    # datasets = get_fold_datasets(my_config['data'])
    # model = training_utils.get_timm_model(my_config['model'])

    training_output = run_training(my_config)
    save_json(training_output, os.path.join(my_config['output_dir'], 'training_output.json'))

    test_acc_sum = 0
    test_acc_weight = 0
    for fold_out in training_output:
        weight = len(fold_out['data_split']['test'])
        test_acc_sum += weight * fold_out['history'][-1]['val_accuracy']
        test_acc_weight += weight
    test_acc = test_acc_sum / test_acc_weight
    print(f'overal test accuracy: {test_acc:0.4f}')

