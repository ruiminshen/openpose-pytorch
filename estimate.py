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

import argparse
import configparser
import logging
import logging.config
import os
import time
import re
import yaml

import numpy as np
import torch.autograd
import torch.cuda
import torch.optim
import torch.utils.data
import torch.nn as nn
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
import humanize
import pybenchmark
import cv2

import transform
import model
import utils.train
import utils.visualize
import pyopenpose


class Estimate(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.cache_dir = utils.get_cache_dir(config)
        self.model_dir = utils.get_model_dir(config)
        _, self.num_parts = utils.get_dataset_mappers(config)
        self.limbs_index = utils.get_limbs_index(config)
        if args.debug is None:
            self.draw_cluster = utils.visualize.DrawCluster(colors=args.colors, thickness=args.thickness)
        else:
            self.draw_feature = utils.visualize.DrawFeature()
            s = re.search('(-?[0-9]+)([a-z]+)(-?[0-9]+)', args.debug)
            stage = int(s.group(1))
            name = s.group(2)
            channel = int(s.group(3))
            self.get_feature = lambda outputs: outputs[stage][name][0][channel]
        self.height, self.width = tuple(map(int, config.get('image', 'size').split()))
        if args.caffe:
            init_net = caffe2_pb2.NetDef()
            with open(os.path.join(self.model_dir, 'init_net.pb'), 'rb') as f:
                init_net.ParseFromString(f.read())
            predict_net = caffe2_pb2.NetDef()
            with open(os.path.join(self.model_dir, 'predict_net.pb'), 'rb') as f:
                predict_net.ParseFromString(f.read())
            p = workspace.Predictor(init_net, predict_net)
            self.inference = lambda tensor: [{'parts': torch.autograd.Variable(torch.from_numpy(parts)), 'limbs': torch.autograd.Variable(torch.from_numpy(limbs))} for parts, limbs in zip(*[iter(p.run([tensor.data.cpu().numpy()]))] * 2)]
        else:
            self.step, self.epoch, self.dnn, self.stages = self.load()
            self.inference = model.Inference(config, self.dnn, self.stages)
            self.inference.eval()
            if torch.cuda.is_available():
                self.inference.cuda()
            logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in self.inference.state_dict().values())))
        self.cap = self.create_cap()
        self.keys = set(args.keys)
        self.resize = transform.parse_transform(config, config.get('transform', 'resize_test'))
        self.transform_image = transform.get_transform(config, config.get('transform', 'image_test').split())
        self.transform_tensor = transform.get_transform(config, config.get('transform', 'tensor').split())

    def __del__(self):
        cv2.destroyAllWindows()
        try:
            self.writer.release()
        except AttributeError:
            pass
        self.cap.release()

    def load(self):
        path, step, epoch = utils.train.load_model(self.model_dir)
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        config_channels_dnn = model.ConfigChannels(self.config, state_dict['dnn'])
        dnn = utils.parse_attr(self.config.get('model', 'dnn'))(config_channels_dnn)
        config_channels_stages = model.ConfigChannels(self.config, state_dict['stages'], config_channels_dnn.channels)
        channel_dict = model.channel_dict(self.num_parts, len(self.limbs_index))
        stages = nn.Sequential(*[utils.parse_attr(s)(config_channels_stages, channel_dict, config_channels_dnn.channels, str(i)) for i, s in enumerate(self.config.get('model', 'stages').split())])
        dnn.load_state_dict(config_channels_dnn.state_dict)
        stages.load_state_dict(config_channels_stages.state_dict)
        return step, epoch, dnn, stages

    def create_cap(self):
        try:
            cap = int(self.args.input)
        except ValueError:
            cap = os.path.expanduser(os.path.expandvars(self.args.input))
            assert os.path.exists(cap)
        return cv2.VideoCapture(cap)

    def create_writer(self, height, width):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logging.info('cap fps=%f' % fps)
        path = os.path.expanduser(os.path.expandvars(self.args.output))
        if self.args.fourcc:
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc.upper())
        else:
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return cv2.VideoWriter(path, fourcc, fps, (width, height))

    def get_image(self):
        ret, image_bgr = self.cap.read()
        if self.args.crop:
            image_bgr = image_bgr[self.crop_ymin:self.crop_ymax, self.crop_xmin:self.crop_xmax]
        return image_bgr

    def __call__(self):
        image_bgr = self.get_image()
        image_resized = self.resize(image_bgr, self.height, self.width)
        image = self.transform_image(image_resized)
        tensor = self.transform_tensor(image)
        tensor = utils.ensure_device(tensor.unsqueeze(0))
        outputs = pybenchmark.profile('inference')(self.inference)(torch.autograd.Variable(tensor, volatile=True))
        if hasattr(self, 'draw_cluster'):
            output = outputs[-1]
            parts, limbs = (output[name][0].data for name in 'parts, limbs'.split(', '))
            parts = parts[:-1]
            parts, limbs = (t.cpu().numpy() for t in (parts, limbs))
            try:
                interpolation = getattr(cv2, 'INTER_' + self.config.get('estimate', 'interpolation').upper())
                parts, limbs = (np.stack([cv2.resize(feature, (self.width, self.height), interpolation=interpolation) for feature in a]) for a in (parts, limbs))
            except configparser.NoOptionError:
                pass
            clusters = pyopenpose.estimate(
                parts, limbs,
                self.limbs_index,
                self.config.getfloat('nms', 'threshold'),
                self.config.getfloat('integration', 'step'), tuple(map(int, self.config.get('integration', 'step_limits').split())), self.config.getfloat('integration', 'min_score'), self.config.getint('integration', 'min_count'),
                self.config.getfloat('cluster', 'min_score'), self.config.getint('cluster', 'min_count'),
            )
            scale_y, scale_x = self.resize.scale(parts.shape[-2:], image_bgr.shape[:2])
            image_result = image_bgr.copy()
            for cluster in clusters:
                cluster = [((i1, int(y1 * scale_y), int(x1 * scale_x)), (i2, int(y2 * scale_y), int(x2 * scale_x))) for (i1, y1, x1), (i2, y2, x2) in cluster]
                image_result = self.draw_cluster(image_result, cluster)
        else:
            image_result = image_resized.copy()
            feature = self.get_feature(outputs).data.cpu().numpy()
            image_result = self.draw_feature(image_result, feature)
        if self.args.output:
            if not hasattr(self, 'writer'):
                self.writer = self.create_writer(*image_result.shape[:2])
            self.writer.write(image_result)
        else:
            cv2.imshow('estimate', image_result)
        if cv2.waitKey(0 if self.args.pause else 1) in self.keys:
            root = os.path.join(self.model_dir, 'snapshot')
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, time.strftime(self.args.format))
            cv2.imwrite(path, image_bgr)
            logging.warning('image dumped into ' + path)


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    detect = Estimate(args, config)
    try:
        while detect.cap.isOpened():
            detect()
    except KeyboardInterrupt:
        logging.warning('interrupted')
    finally:
        logging.info(pybenchmark.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-i', '--input', default=-1)
    parser.add_argument('-k', '--keys', nargs='+', type=int, default=[ord(' ')], help='keys to dump images')
    parser.add_argument('-o', '--output', help='output video file')
    parser.add_argument('-f', '--format', default='%Y-%m-%d_%H-%M-%S.jpg', help='dump file name format')
    parser.add_argument('--crop', nargs='+', type=float, default=[], help='ymin ymax xmin xmax')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--fourcc', default='XVID', help='4-character code of codec used to compress the frames, such as XVID, MJPG')
    parser.add_argument('--thickness', default=3, type=int)
    parser.add_argument('--colors', nargs='+', default=[])
    parser.add_argument('-d', '--debug')
    parser.add_argument('--caffe', action='store_true')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
