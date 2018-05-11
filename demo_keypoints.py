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
import argparse
import configparser
import logging
import logging.config
import yaml

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import utils.data
import utils.visualize


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
    mask_ext = config.get('cache', 'mask_ext')
    paths = [os.path.join(cache_dir, phase + '.pkl') for phase in args.phase]
    dataset = utils.data.Dataset(config, utils.data.load_pickles(paths))
    logging.info('num_examples=%d' % len(dataset))
    draw_points = utils.visualize.DrawPoints(limbs_index, colors=config.get('draw_points', 'colors').split())
    draw_bbox = utils.visualize.DrawBBox(config)
    for data in dataset:
        path, keypath, keypoints, yx_min, yx_max = (data[key] for key in 'path, keypath, keypoints, yx_min, yx_max'.split(', '))
        image = scipy.misc.imread(path, mode='RGB')
        fig = plt.figure()
        ax = fig.gca()
        maskpath = keypath + '.mask' + mask_ext
        mask = scipy.misc.imread(maskpath)
        image = utils.visualize.draw_mask(image, mask)
        for points in keypoints:
            image = draw_points(image, points)
        image = draw_bbox(image, yx_min.astype(np.int), yx_max.astype(np.int))
        ax.imshow(image)
        ax.set_xlim([0, image.shape[1] - 1])
        ax.set_ylim([image.shape[0] - 1, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-p', '--phase', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()

if __name__ == '__main__':
    main()
