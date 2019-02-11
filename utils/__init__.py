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

import os
import re
import configparser
import importlib
import hashlib

import numpy as np
import pandas as pd
import torch.autograd
from PIL import Image

import pyopenpose


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, yx_min, yx_max, cls):
        for t in self.transforms:
            img, yx_min, yx_max, cls = t(img, yx_min, yx_max, cls)
        return img, yx_min, yx_max, cls


class RegexList(list):
    def __init__(self, l):
        for s in l:
            prog = re.compile(s)
            self.append(prog)

    def __call__(self, s):
        for prog in self:
            if prog.match(s):
                return True
        return False


class DatasetMapper(object):
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, parts, dtype=np.int64):
        assert len(parts.shape) == 2 and parts.shape[-1] == 3
        result = np.zeros([len(self.mapper), 3], dtype=parts.dtype)
        for i, func in enumerate(self.mapper):
            result[i] = func(parts)
        return result


def get_dataset_mappers(config):
    root = os.path.expanduser(os.path.expandvars(config.get('cache', 'dataset')))
    mappers = {}
    for dataset in os.listdir(root):
        path = os.path.join(root, dataset)
        if os.path.isfile(path):
            with open(path, 'r') as f:
                mapper = [eval(line.rstrip()) for line in f]
            mappers[dataset] = mapper
    sizes = set(map(lambda mapper: len(mapper), mappers.values()))
    assert len(sizes) == 1
    for dataset in mappers:
        mappers[dataset] = DatasetMapper(mappers[dataset])
    return mappers, next(iter(sizes))


def get_limbs_index(config):
    dataset = os.path.expanduser(os.path.expandvars(config.get('cache', 'dataset')))
    limbs_index = np.loadtxt(dataset + '.tsv', dtype=np.int, delimiter='\t', ndmin=2)
    if len(limbs_index) > 0:
        assert pyopenpose.limbs_points(limbs_index) == get_dataset_mappers(config)[1]
    else:
        limbs_index = np.reshape(limbs_index, [0, 2])
    return limbs_index


def get_cache_dir(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    name = config.get('cache', 'name')
    dataset = os.path.basename(config.get('cache', 'dataset'))
    return os.path.join(root, name, dataset)


def get_model_dir(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    name = config.get('model', 'name')
    dataset = os.path.basename(config.get('cache', 'dataset'))
    dnn = config.get('model', 'dnn')
    stages = hashlib.md5(' '.join(config.get('model', 'stages').split()).encode()).hexdigest()
    return os.path.join(root, name, dataset, dnn, stages)


def get_eval_db(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    db = config.get('eval', 'db')
    return os.path.join(root, db)


def get_category(config, cache_dir=None):
    path = os.path.expanduser(os.path.expandvars(config.get('cache', 'category'))) if cache_dir is None else os.path.join(cache_dir, 'category')
    with open(path, 'r') as f:
        return [line.strip() for line in f]


def get_anchors(config, dtype=np.float32):
    path = os.path.expanduser(os.path.expandvars(config.get('model', 'anchors')))
    df = pd.read_csv(path, sep='\t', dtype=dtype)
    return df[['height', 'width']].values


def parse_attr(s):
    m, n = s.rsplit('.', 1)
    m = importlib.import_module(m)
    return getattr(m, n)


def load_config(config, paths):
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        assert os.path.exists(path)
        config.read(path)


def modify_config(config, cmd):
    var, value = cmd.split('=', 1)
    section, option = var.split('/')
    if value:
        config.set(section, option, value)
    else:
        try:
            config.remove_option(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass


def dense(var):
    return [torch.mean(torch.abs(x)) if torch.is_tensor(x) else np.abs(x) for x in var]


def abs_mean(data, dtype=np.float32):
    assert isinstance(data, np.ndarray), type(data)
    return np.sum(np.abs(data)) / dtype(data.size)


def image_size(path):
    with Image.open(path) as image:
        return image.size
