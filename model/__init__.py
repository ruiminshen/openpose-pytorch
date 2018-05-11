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

import logging
import collections

import torch
import torch.nn as nn
import torch.autograd


class ConfigChannels(object):
    def __init__(self, config, state_dict=None, channels=3):
        self.config = config
        self.state_dict = state_dict
        self.channels = channels

    def __call__(self, default, name, fn=lambda var: var.size(0)):
        if self.state_dict is None:
            self.channels = default
        else:
            var = self.state_dict[name]
            self.channels = fn(var)
            if self.channels != default:
                logging.warning('%s: change number of output channels from %d to %d' % (name, default, self.channels))
        return self.channels


def channel_dict(num_parts, num_limbs):
    return collections.OrderedDict([
        ('parts', num_parts + 1),
        ('limbs', num_limbs * 2),
    ])


def feature_size(dnn, height, width):
    image = torch.autograd.Variable(torch.randn(1, 3, height, width), volatile=True)
    if next(dnn.parameters()).is_cuda:
        image = image.cuda()
    feature = dnn(image)
    return feature.size()[-2:]


class Inference(nn.Module):
    def __init__(self, config, dnn, stages):
        nn.Module.__init__(self)
        self.config = config
        self.dnn = dnn
        self.stages = stages

    def forward(self, x):
        x = self.dnn(x)
        outputs = []
        output = {}
        for stage in self.stages:
            output = stage(x, **output)
            outputs.append(output)
        return outputs


class Loss(object):
    def __init__(self, config, data, limbs_index, height, width):
        self.config = config
        self.data = data
        self.limbs_index = limbs_index
        self.height = height
        self.width = width

    def __call__(self, **kwargs):
        mask = torch.autograd.Variable(self.data['mask'].float())
        batch_size, rows, cols = mask.size()
        mask = mask.view(batch_size, 1, rows, cols)
        data = {name: torch.autograd.Variable(self.data[name]) for name in kwargs}
        return {name: self.loss(mask, data[name], feature) for name, feature in kwargs.items()}

    def loss(self, mask, label, feature):
        return torch.mean(mask * (feature - label) ** 2)
