# -*- coding: utf-8 -*-
import torch


class ChannelDenseStem(torch.nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        act_layer = torch.nn.ReLU
        norm_layer = torch.nn.BatchNorm2d
        layers = []
        for in_channels, out_channels in zip(channels[:-2], channels[1:-1]):
            layers += [
                torch.nn.Conv2d(in_channels, out_channels, (1,1)),
                norm_layer(out_channels),
                act_layer(inplace=True),
                torch.nn.Dropout(dropout)
                ]
        layers += [torch.nn.Conv2d(channels[-2], channels[-1], (1,1))]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def timm_deep_stem(in_chans, out_chans, stem_type='deep', stem_width=64):
    stem_chs = (stem_width, stem_width)
    act_layer = torch.nn.ReLU
    norm_layer = torch.nn.BatchNorm2d
    if 'tiered' in stem_type:
        stem_chs = (3 * (stem_width // 4), stem_width)
    stem = torch.nn.Sequential(*[
        torch.nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
        norm_layer(stem_chs[0]),
        act_layer(inplace=True),
        torch.nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
        norm_layer(stem_chs[1]),
        act_layer(inplace=True),
        torch.nn.Conv2d(stem_chs[1], out_chans, 3, stride=1, padding=1, bias=False)])
    return stem



#%%
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

def replace_first_layer(model, replace_genrator):
    """

    replace_genrator : function: number of channels (int) -> layer

    """
    first_layer = get_first_layer(model)
    channels_out = first_layer.weight.shape[0]
    new_first_layer = replace_genrator(channels_out)
    set_first_layer(model, new_first_layer)

def add_first_layer(model, layer):
    old_first_layer = get_first_layer(model)
    new_first_layer = torch.nn.Sequential(layer, old_first_layer)
    set_first_layer(model, new_first_layer)


if __name__ == '__main__':
    import timm
    torch.set_grad_enabled(False)

    resnet = timm.create_model('resnet18')
    densenet = timm.create_model('densenet121')
    inputs = torch.rand(2,1024,32,32)
    for model in [resnet, densenet]:
        replace_first_layer(model, lambda x: ChannelDenseStem([1024, 256, 128, x]))
        out = model(inputs)
