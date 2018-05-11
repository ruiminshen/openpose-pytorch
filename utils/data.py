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
import pickle
import random
import copy

import numpy as np
import torch.utils.data
import cv2

import utils
import pyopenpose


def padding_labels(data, dim, labels='keypoints, yx_min, yx_max'.split(', ')):
    """
    Padding labels into the same dimension (to form a batch).
    :author 申瑞珉 (Ruimin Shen)
    :param data: A dict contains the labels to be padded.
    :param dim: The target dimension.
    :param labels: The list of label names.
    :return: The padded label dict.
    """
    pad = dim - len(data[labels[0]])
    for key in labels:
        label = data[key]
        data[key] = np.pad(label, [(0, pad)] + [(0, 0)] * (len(label.shape) - 1), 'constant')
    return data


def load_pickles(paths):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data += pickle.load(f)
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, data, transform=lambda data: data, shuffle=False, dir=None):
        """
        Load the cached data (.pkl) into memory.
        :author 申瑞珉 (Ruimin Shen)
        :param data: A list contains the data samples (dict).
        :param transform: A function transforms (usually performs a sequence of data augmentation operations) the labels in a dict.
        :param shuffle: Shuffle the loaded dataset.
        :param dir: The directory to store the exception data.
        """
        self.config = config
        self.mask_ext = config.get('cache', 'mask_ext')
        self.data = data
        if shuffle:
            random.shuffle(self.data)
        self.transform = transform
        self.dir = dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        try:
            image = cv2.imread(data['path'])
            data['image'] = image
            data['size'] = np.array(image.shape[:2])
            mask = cv2.imread(data['keypath'] + '.mask' + self.mask_ext, cv2.IMREAD_GRAYSCALE)
            assert image.shape[:2] == mask.shape, [image.shape[:2], mask.shape]
            data['mask'] = mask
            data['index'] = random.randint(0, len(data['keypoints']) - 1)
            data = self.transform(data)
        except:
            if self.dir is not None:
                os.makedirs(self.dir, exist_ok=True)
                name = self.__module__ + '.' + type(self).__name__
                with open(os.path.join(self.dir, name + '.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            raise
        return data


class Collate(object):
    def __init__(self, config, resize, sizes, feature_sizes, maintain=1, transform_image=lambda image: image, transform_tensor=None, dir=None):
        """
        Unify multiple data samples (e.g., resize images into the same size, and padding bounding box labels into the same number) to form a batch.
        :author 申瑞珉 (Ruimin Shen)
        :param resize: A function to resize the image and labels.
        :param sizes: The image sizes to be randomly choosed.
        :param feature_sizes: The feature sizes related to the image sizes.
        :param maintain: How many times a size to be maintained.
        :param transform_image: A function to transform the resized image.
        :param transform_tensor: A function to standardize a image into a tensor.
        :param dir: The directory to store the exception data.
        """
        self.config = config
        self.resize = resize
        assert len(sizes) == len(feature_sizes)
        self.sizes = sizes
        self.feature_sizes = feature_sizes
        assert maintain > 0
        self.maintain = maintain
        self._maintain = maintain
        self.transform_image = transform_image
        self.transform_tensor = transform_tensor
        self.dir = dir
        self.sigma_parts = config.getfloat('label', 'sigma_parts')
        self.sigma_limbs = config.getfloat('label', 'sigma_limbs')
        self.limbs_index = utils.get_limbs_index(config)

    def __call__(self, batch):
        (height, width), (rows, cols) = self.next_size()
        dim = max(len(data['keypoints']) for data in batch)
        _batch = []
        for data in batch:
            try:
                data = self.resize(data, height, width)
                data['image'] = self.transform_image(data['image'])
                data = padding_labels(data, dim)
                if self.transform_tensor is not None:
                    data['tensor'] = self.transform_tensor(data['image'])
                data['mask'] = (cv2.resize(data['mask'], (cols, rows)) > 127).astype(np.uint8)
                data['parts'] = pyopenpose.label_parts(data['keypoints'], self.sigma_parts, height, width, rows, cols)
                data['limbs'] = pyopenpose.label_limbs(data['keypoints'], self.limbs_index, self.sigma_limbs, height, width, rows, cols)
                _batch.append(data)
            except:
                if self.dir is not None:
                    os.makedirs(self.dir, exist_ok=True)
                    name = self.__module__ + '.' + type(self).__name__
                    with open(os.path.join(self.dir, name + '.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                raise
        return torch.utils.data.dataloader.default_collate(_batch)

    def next_size(self):
        if self._maintain < self.maintain:
            self._maintain += 1
        else:
            self._index = random.randint(0, len(self.sizes) - 1)
            self._maintain = 0
        return self.sizes[self._index], self.feature_sizes[self._index]
