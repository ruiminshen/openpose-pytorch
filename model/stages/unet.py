"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import collections.abc

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=True, stride=1, bn=False, act=True):
        nn.Module.__init__(self)
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else lambda x: x
        self.act = nn.ReLU(inplace=True) if act else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bn=False, act=True):
        nn.Module.__init__(self)
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else lambda x: x
        self.act = nn.ReLU(inplace=True) if act else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    def __init__(self, config_channels, channels, prefix, kernel_sizes, pooling):
        nn.Module.__init__(self)
        self.seq = nn.Sequential(*[Conv2d(config_channels.channels, config_channels(channels, '%s.seq.%d.conv.weight' % (prefix, index)), kernel_size) for index, kernel_size in enumerate(kernel_sizes)])
        self.downsample = nn.MaxPool2d(kernel_size=pooling)

    def forward(self, x):
        feature = self.seq(x)
        return self.downsample(feature), feature


class Upsample(nn.Module):
    def __init__(self, config_channels, channels, channels_min, prefix, sample, kernel_sizes, ratio=1):
        nn.Module.__init__(self)
        self.upsample = ConvTranspose2d(config_channels.channels, config_channels(channels, '%s.upsample.conv.weight' % prefix, fn=lambda var: var.size(1)), kernel_size=sample, stride=sample)
        config_channels.channels += channels  # concat

        seq = []
        if ratio < 1:
            seq.append(Conv2d(config_channels.channels, config_channels(max(int(config_channels.channels * ratio), channels_min), '%s.seq.%d.conv.weight' % (prefix, len(seq))), 1))
        for kernel_size in kernel_sizes:
            seq.append(Conv2d(config_channels.channels, config_channels(channels, '%s.seq.%d.conv.weight' % (prefix, len(seq))), kernel_size))
        self.seq = nn.Sequential(*seq)

    def forward(self, x, feature):
        x = self.upsample(x)
        x = torch.cat([x, feature], 1)
        return self.seq(x)


class Branch(nn.Module):
    def __init__(self, config_channels, channels, prefix, multiply, ratio, kernel_sizes, sample):
        nn.Module.__init__(self)
        _channels = channels
        self.down = []
        for index, m in enumerate(multiply):
            name = 'down%d' % index
            block = Downsample(config_channels, _channels, '%s.%s' % (prefix, name), kernel_sizes, pooling=sample)
            setattr(self, name, block)
            self.down.append(block)
            _channels = int(_channels * m)
        self.top = nn.Sequential(*[Conv2d(config_channels.channels, config_channels(_channels, '%s.top.%d.conv.weight' % (prefix, index)), kernel_size) for index, kernel_size in enumerate(kernel_sizes)])

        self.up = []
        for index, block in enumerate(self.down[::-1]):
            name = 'up%d' % index
            block = Upsample(config_channels, block.seq[-1].conv.weight.size(0), channels, '%s.%s' % (prefix, name), sample, kernel_sizes, ratio)
            setattr(self, name, block)
            self.up.append(block)
        self.out = Conv2d(config_channels.channels, channels, 1, act=False)

    def forward(self, x):
        features = []
        for block in self.down:
            x, feature = block(x)
            features.append(feature)
        x = self.top(x)

        for block, feature in zip(self.up, features[::-1]):
            x = block(x, feature)
        return self.out(x)


class Unet(nn.Module):
    def __init__(self, config_channels, channel_dict, channels_dnn, prefix, multiply=[2, 2], ratio=1, kernel_sizes=[3], sample=2):
        nn.Module.__init__(self)
        channels_stage = config_channels.channels
        for name, channels in channel_dict.items():
            config_channels.channels = channels_stage
            branch = Branch(config_channels, channels, '%s.%s' % (prefix, name), multiply, ratio, kernel_sizes, sample)
            setattr(self, name, branch)
        config_channels.channels = channels_dnn + sum(branch.out.conv.weight.size(0) for branch in self._modules.values())

    def forward(self, x, **kwargs):
        if kwargs:
            x = torch.cat([kwargs[name] for name in ('parts', 'limbs') if name in kwargs] + [x], 1)
        return {name: branch(x) for name, branch in self._modules.items()}


class Unet1Sqz3(Unet):
    def __init__(self, config_channels, channel_dict, channels_dnn, prefix):
        Unet.__init__(self, config_channels, channel_dict, channels_dnn, prefix, multiply=[2], ratio=1 / 3)


class Unet1Sqz3_a(Unet):
    def __init__(self, config_channels, channel_dict, channels_dnn, prefix):
        Unet.__init__(self, config_channels, channel_dict, channels_dnn, prefix, multiply=[1.5], ratio=1 / 3)


class Unet2Sqz3(Unet):
    def __init__(self, config_channels, channel_dict, channels_dnn, prefix):
        Unet.__init__(self, config_channels, channel_dict, channels_dnn, prefix, multiply=[2, 2], ratio=1 / 3)
