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

import inspect

import inflection
import numpy as np
import cv2


def rescale(image, mask, keypoints, yx_min, yx_max, height, width):
    _height, _width = image.shape[:2]
    scale = np.array([height / _height, width / _width], np.float32)
    image = cv2.resize(image, (width, height))
    mask = cv2.resize(mask, (width, height))
    keypoints[:, :, :2] *= scale
    yx_min *= scale
    yx_max *= scale
    return image, mask, keypoints, yx_min, yx_max


class Rescale(object):
    def __init__(self):
        self.fn = eval(inflection.underscore(type(self).__name__))

    def __call__(self, data, height, width):
        data['image'], data['mask'], data['keypoints'], data['yx_min'], data['yx_max'] = self.fn(data['image'], data['mask'], data['keypoints'], data['yx_min'], data['yx_max'], height, width)
        return data


def padding(image, mask, keypoints, yx_min, yx_max, height, width):
    _height, _width, _ = image.shape
    if _height / _width > height / width:
        scale = height / _height
    else:
        scale = width / _width
    m = np.eye(2, 3)
    m[0, 0] = scale
    m[1, 1] = scale
    flags = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    image = cv2.warpAffine(image, m, (width, height), flags=flags)
    mask = cv2.warpAffine(mask, m, (width, height), flags=flags)
    return image, mask, keypoints, yx_min, yx_max


class Padding(object):
    def __init__(self):
        self.fn = eval(inflection.underscore(type(self).__name__))

    def __call__(self, data, height, width):
        data['image'], data['mask'], data['keypoints'], data['yx_min'], data['yx_max'] = self.fn(data['image'], data['mask'], data['keypoints'], data['yx_min'], data['yx_max'], height, width)
        return data


def resize(config, image, mask, keypoints, yx_min, yx_max, height, width):
    fn = eval(config.get('data', inspect.stack()[0][3]))
    return fn(image, mask, keypoints, yx_min, yx_max, height, width)


class Resize(object):
    def __init__(self, config):
        self.config = config
        self.fn = eval(config.get('data', inflection.underscore(type(self).__name__)))

    def __call__(self, data, height, width):
        data['image'], data['yx_min'], data['yx_max'] = self.fn(self.config, data['image'], data['yx_min'], data['yx_max'], height, width)
        return data


def change_aspect_ratio(range, height_src, width_src, height_dst, width_dst):
    assert range >= 0
    if width_src < height_src:
        width = min(range, width_src)
        height = width * height_dst / width_dst
    else:
        height = min(range, height_src)
        width = height * width_dst / height_dst
    return height, width


def repair(yx_min, yx_max, size):
    move = np.clip(yx_max - size, 0, None)
    yx_min -= move
    yx_max -= move
    move = np.clip(-yx_min, 0, None)
    yx_min += move
    yx_max += move
    return yx_min, yx_max


def random_crop(config, image, mask, keypoints, yx_min, yx_max, index, height, width):
    name = inspect.stack()[0][3]
    scale1, scale2 = tuple(map(float, config.get('augmentation', name).split()))
    assert 1 <= scale1 <= scale2, (scale1, scale2)
    dtype = keypoints.dtype
    size = np.array(image.shape[:2], dtype)
    _yx_min, _yx_max = yx_min[index], yx_max[index]
    _center = (_yx_min + _yx_max) / 2
    _size = np.array(change_aspect_ratio(np.max(_yx_max - _yx_min), *size, height, width), dtype)
    _size1, _size2 = _size * scale1 / 2, _size * scale2 / 2
    yx_min1, yx_max1 = _center - _size1, _center + _size1
    yx_min2, yx_max2 = _center - _size2, _center + _size2
    yx_min1, yx_max1 = repair(yx_min1, yx_max1, size)
    yx_min2, yx_max2 = repair(yx_min2, yx_max2, size)
    margin = np.random.rand(4).astype(dtype) * np.concatenate([yx_min1 - yx_min2, yx_max2 - yx_max1], 0)
    yx_min_crop = np.clip(yx_min2 + margin[:2], 0, None)
    yx_max_crop = np.clip(yx_max2 - margin[2:], None, size)
    _ymin, _xmin = tuple(map(int, yx_min_crop))
    _ymax, _xmax = tuple(map(int, yx_max_crop))
    image = image[_ymin:_ymax, _xmin:_xmax, :]
    mask = mask[_ymin:_ymax, _xmin:_xmax]
    keypoints[:, :, :2] -= yx_min_crop
    yx_min -= yx_min_crop
    yx_max -= yx_min_crop
    return rescale(image, mask, keypoints, yx_min, yx_max, height, width)


class RandomCrop(object):
    def __init__(self, config):
        self.config = config
        self.fn = eval(inflection.underscore(type(self).__name__))

    def __call__(self, data, height, width):
        data['image'], data['mask'], data['keypoints'], data['yx_min'], data['yx_max'] = self.fn(self.config, data['image'], data['mask'], data['keypoints'], data['yx_min'], data['yx_max'], data['index'], height, width)
        return data
