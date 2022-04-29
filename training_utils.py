# -*- coding: utf-8 -*-
import os

import torch
import timm

#%%
class CheckpointSaver:
    def __init__(self, get_save_path, device, max_saved=5):
        """
        get_save_path: function from score to file path
        device: 'cpu' or 'cuda'
        max_saved: number of best weights to save
        """
        self.best = None
        self.device = device

        self.get_save_path = get_save_path
        self.max_saved = max_saved
        self.saved = []

    def save_if_best(self, model, score):
        if self.best is None or score>=self.best:
            self.best = score
            output_path = self.get_save_path(score)
            torch.save(model.cpu().state_dict(), output_path)
            if len(self.saved) == 0 or self.saved[-1] != output_path:
                self.saved.append(output_path)
                if len(self.saved) > self.max_saved:
                    os.remove(self.saved.pop(0))
            model.to(self.device)
            return True
        return False

#%%

def get_timm_model(model_config):

    if model_config.get('linear_channels_adapter', False):
        timm_in_chans = 3
    else:
        timm_in_chans = model_config['in_chans']
    model = timm.create_model(model_config['name'],
                              num_classes = model_config['num_classes'],
                              in_chans = timm_in_chans,
                              pretrained = model_config['pretrained_timm'] and not model_config['pretrained_own'], # load timm weights if requested and if no custom weights will be used
                              **model_config['timm_kwargs'])

    # add layer at the beginning of the network
    first_layer = [get_first_layer(model)]
    if model_config.get('linear_channels_adapter', False):
        first_layer.insert(0, torch.nn.Conv2d(model_config['in_chans'], 3, 1, bias=False))
    set_first_layer(model, torch.nn.Sequential(*first_layer)) #model.conv1 = torch.nn.Sequential(*first_layer)

    # load weights from file
    if model_config['pretrained_own']:
        model.load_state_dict(torch.load(model_config['weights_path']))

    return model

def get_first_layer(model):
    stages = list(model.state_dict().keys())[0].split('.')[:-1]
    out = model
    for s in stages:
        if s.isnumeric():
            out = out[int(s)]
        else:
            out = getattr(out, s)
    return out

def set_first_layer(model, layer):
    stages = list(model.state_dict().keys())[0].split('.')[:-1]

    parent_module = model
    for s in stages[:-1]:
        s = stages[0]
        if s.isnumeric():
            parent_module = parent_module[int(s)]
        else:
            parent_module = getattr(parent_module, s)

    s = stages[-1]

    if s.isnumeric():
        parent_module[int(s)] = layer
    else:
        setattr(parent_module, s, layer)

    return model

#%%

def get_optimizer_and_scheduler(model, optimizer_config, scheduler_config):
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = getattr(torch.optim, optimizer_config['type'])(params,
                                                               **optimizer_config['kwargs'])
    base_scheduler = getattr(torch.optim.lr_scheduler, scheduler_config['type'])(optimizer,
                                                                            **scheduler_config['kwargs'])
    if scheduler_config['warm-up_epochs'] in [0, None]:
        scheduler = base_scheduler
    else:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=scheduler_config['warm-up_epochs'], after_scheduler=base_scheduler)

    return optimizer, scheduler

#%%

def get_criterion(criterion_config):
    if criterion_config['type'] == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    raise NotImplementedError()

#%%
"""
https://github.com/ildoonet/pytorch-gradual-warmup-lr
    optim = SGD(model, 0.1)
    scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
"""

class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if not isinstance(self.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)