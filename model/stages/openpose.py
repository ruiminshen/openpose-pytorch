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


class Stage0(nn.Module):
    def __init__(self, config_channels, channel_dict, channels_dnn, prefix):
        nn.Module.__init__(self)
        channels_stage = config_channels.channels
        for name, channels in channel_dict.items():
            config_channels.channels = channels_stage
            branch = []
            for _ in range(3):
                branch.append(Conv2d(config_channels.channels, config_channels(128, '%s.%s.%d.conv.weight' % (prefix, name, len(branch))), 3))
            branch.append(Conv2d(config_channels.channels, config_channels(512, '%s.%s.%d.conv.weight' % (prefix, name, len(branch))), 1))
            branch.append(Conv2d(config_channels.channels, channels, 1, act=False))
            setattr(self, name, nn.Sequential(*branch))
        config_channels.channels = channels_dnn + sum(branch[-1].conv.weight.size(0) for branch in self._modules.values())
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.zero_()

    def forward(self, x, **kwargs):
        return {name: var(x) for name, var in self._modules.items()}


class Stage(nn.Module):
    def __init__(self, config_channels, channels, channels_dnn, prefix):
        nn.Module.__init__(self)
        channels_stage = config_channels.channels
        for name, _channels in channels.items():
            config_channels.channels = channels_stage
            branch = []
            for _ in range(5):
                branch.append(Conv2d(config_channels.channels, config_channels(128, '%s.%s.%d.conv.weight' % (prefix, name, len(branch))), 7))
            branch.append(Conv2d(config_channels.channels, config_channels(128, '%s.%s.%d.conv.weight' % (prefix, name, len(branch))), 1))
            branch.append(Conv2d(config_channels.channels, _channels, 1, act=False))
            setattr(self, name, nn.Sequential(*branch))
        config_channels.channels = channels_dnn + sum(branch[-1].conv.weight.size(0) for branch in self._modules.values())
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.zero_()

    def forward(self, x, **kwargs):
        x = torch.cat([kwargs[name] for name in ('limbs', 'parts')] + [x], 1)
        return {name: var(x) for name, var in self._modules.items()}
