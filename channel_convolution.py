# -*- coding: utf-8 -*-
from typing import Union, Tuple
import numpy as np
from torch import nn

def conv_output_size(size, kernel, stride, padding, dilation):
    result = (size + 2*padding -dilation*(kernel-1)-1)/stride + 1
    return int(np.floor(result))


class ChannelConv(nn.Module):
    def __init__(self,
                out_channels: int,
                kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1,
                padding: Union[int, Tuple[int]] = 0,
                dilation: Union[int, Tuple[int]] = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros'
                ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(kernel_size,1,1),
            stride=(stride, 1, 1),
            padding=(padding,0,0),
            dilation=(dilation,1,1),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
            )

    def forward(self, x):
        return self.conv(x.unsqueeze(1))

