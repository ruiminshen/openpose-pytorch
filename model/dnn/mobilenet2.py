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

import torch.nn as nn
import math


def conv_bn(inp, oup, stride, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, dilation=dilation),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet2(nn.Module):
    def __init__(self, config_channels, input_size=224, last_channel=320, width_mult=1., dilation=1, ratio=1):
        nn.Module.__init__(self)
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, int(16 * ratio), 1, 1],
            [6, int(24 * ratio), 2, 2],
            [6, int(32 * ratio), 3, 2],
            [6, int(64 * ratio), 4, 1], # stride 2->1
            [6, int(96 * ratio), 3, 1],
            [6, int(160 * ratio), 3, 1], # stride 2->1
            [6, int(320 * ratio), 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        if last_channel is None:
            self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        else:
            self.last_channel = int(last_channel * ratio)
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_bn(input_channel, self.last_channel, 1, dilation=dilation))
        #self.features.append(nn.AvgPool2d(input_size/32))
        config_channels.channels = self.last_channel # temp

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        return self.features(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNet2Dilate2(MobileNet2):
    def __init__(self, config_channels):
        MobileNet2.__init__(self, config_channels, dilation=2)


class MobileNet2Dilate4(MobileNet2):
    def __init__(self, config_channels):
        MobileNet2.__init__(self, config_channels, dilation=4)


class MobileNet2Half(MobileNet2):
    def __init__(self, config_channels):
        MobileNet2.__init__(self, config_channels, ratio=1 / 2)


class MobileNet2Quarter(MobileNet2):
    def __init__(self, config_channels):
        MobileNet2.__init__(self, config_channels, ratio=1 / 4)
