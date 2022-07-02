# -*- coding: utf-8 -*-

import torch

import timm

import stems

#%%

def get_model(model_config):
    """

    Parameters
    ----------
    model_config : dict
        required keys: name, num_classes, in_chans, stem: {'timm': ..., 'extra': ..., 'extra_kwargs': ...}, pretrained_timm, pretrained_own

    Returns
    -------
    model: pytorch module
    """

    stem_type = model_config['stem']['extra']
    timm_in_chans = model_config['in_chans'] if stem_type == '' else 3
    model = timm.create_model(
        model_config['name'],
        num_classes = model_config['num_classes'],
        in_chans = timm_in_chans, #model_config['in_chans'],
        stem_type = '', # with 'deep' stem type loading of pretrained model fails.
        pretrained = model_config['pretrained_timm'] if not model_config['pretrained_own'] else False,
        **model_config['timm_kwargs']
        )
    if model_config['stem']['timm'] != '':
        stems.replace_first_layer(model, lambda x: stems.timm_deep_stem(model_config['in_chans'], x, stem_type=model_config['stem']['timm'], stem_width=64))


    if stem_type == 'add_linear':
        stems.add_first_layer(model, torch.nn.Conv2d(model_config['in_chans'], 3, 1, bias=False))
    elif stem_type == 'dense':
        intermediate_chans = model_config['stem']['extra_kwargs'].get('intermediate_chans', [256, 128])
        stems.replace_first_layer(model, lambda x: stems.ChannelDenseStem([model_config['in_chans']] + intermediate_chans + [x]))

    if model_config['pretrained_own']:
        model.load_state_dict(torch.load(model_config['weights_path']))

    return model


#%%

def get_optimizer_and_scheduler(model, optimizer_config, scheduler_config):
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = getattr(torch.optim, optimizer_config['type'])(params,
                                                               **optimizer_config['kwargs'])
    base_scheduler = getattr(torch.optim.lr_scheduler, scheduler_config['type'])(optimizer,
                                                                            **scheduler_config['kwargs'])
    if scheduler_config['warm-up_epochs'] is None:
        scheduler = base_scheduler
    else:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=scheduler_config['warm-up_epochs'], after_scheduler=base_scheduler)

    return optimizer, scheduler

#%%

def get_criterion(criterion_config):
    if criterion_config['type'] == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(**criterion_config.get('kwargs', {}))
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