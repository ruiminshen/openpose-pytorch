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

import sys
import os
import argparse
import configparser
import logging
import logging.config
import multiprocessing
import yaml

import numpy as np
import torch.utils.data
from PyQt5 import QtCore, QtWidgets
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg as qtagg
import humanize
import cv2

import model
import utils.data
import utils.train
import utils.visualize
import transform.augmentation


class Visualizer(QtWidgets.QDialog):
    def __init__(self, name, image, feature, alpha=0.5):
        super(Visualizer, self).__init__()
        self.name = name
        self.image = image
        self.feature = feature
        self.draw_feature = utils.visualize.DrawFeature(alpha)
        
        layout = QtWidgets.QVBoxLayout(self)
        fig = plt.Figure()
        self.ax = fig.gca()
        self.canvas = qtagg.FigureCanvasQTAgg(fig)
        layout.addWidget(self.canvas)
        toolbar = qtagg.NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, feature.shape[0] - 1)
        layout.addWidget(self.slider)
        self.slider.valueChanged[int].connect(self.on_progress)
        
        self.ax.imshow(self.image)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.on_progress(0)

    def on_progress(self, index):
        try:
            self.last.remove()
        except AttributeError:
            pass
        image = np.copy(self.image)
        feature = self.feature[index, :, :]
        image = self.draw_feature(image, feature)
        self.last = self.ax.imshow(image)
        self.canvas.draw()
        plt.draw()
        self.setWindowTitle('%s %d/%d' % (self.name, index + 1, self.feature.shape[0]))


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    cache_dir = utils.get_cache_dir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    dnn = utils.parse_attr(config.get('model', 'dnn'))(model.ConfigChannels(config))
    logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in dnn.state_dict().values())))
    size = tuple(map(int, config.get('image', 'size').split()))
    draw_points = utils.visualize.DrawPoints(limbs_index, colors=config.get('draw_points', 'colors').split())
    _draw_points = utils.visualize.DrawPoints(limbs_index, thickness=1)
    draw_bbox = utils.visualize.DrawBBox()
    paths = [os.path.join(cache_dir, phase + '.pkl') for phase in args.phase]
    dataset = utils.data.Dataset(
        config,
        utils.data.load_pickles(paths),
        transform=transform.augmentation.get_transform(config, config.get('transform', 'augmentation').split()),
        shuffle=config.getboolean('data', 'shuffle'),
    )
    logging.info('num_examples=%d' % len(dataset))
    try:
        workers = config.getint('data', 'workers')
    except configparser.NoOptionError:
        workers = multiprocessing.cpu_count()
    collate_fn = utils.data.Collate(
        config,
        transform.parse_transform(config, config.get('transform', 'resize_train')),
        [size], [model.feature_size(dnn, *size)],
        maintain=config.getint('data', 'maintain'),
        transform_image=transform.get_transform(config, config.get('transform', 'image_train').split()),
    )
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    for data in loader:
        path, size, image, mask, keypoints, yx_min, yx_max, parts, limbs, index = (t.numpy() if hasattr(t, 'numpy') else t for t in (data[key] for key in 'path, size, image, mask, keypoints, yx_min, yx_max, parts, limbs, index'.split(', ')))
        for path, size, image, mask, keypoints, yx_min, yx_max, parts, limbs, index in zip(*[path, size, image, mask, keypoints, yx_min, yx_max, parts, limbs, index]):
            logging.info(path + ': ' + 'x'.join(map(str, size)))
            image = utils.visualize.draw_mask(image, mask, 1)
            size = yx_max - yx_min
            target = np.logical_and(*[np.squeeze(a, -1) > 0 for a in np.split(size, size.shape[-1], -1)])
            keypoints, yx_min, yx_max = (a[target] for a in (keypoints, yx_min, yx_max))
            for i, points in enumerate(keypoints):
                if i == index:
                    image = draw_points(image, points)
                else:
                    image = _draw_points(image, points)
            image = draw_bbox(image, yx_min.astype(np.int), yx_max.astype(np.int))
            dialog = Visualizer('parts', image, parts)
            dialog.exec()
            dialog = Visualizer('limbs', image, limbs)
            dialog.exec()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main()
    sys.exit(app.exec_())
