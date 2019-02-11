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
import csv
import hashlib
import shutil
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.autograd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorboardX import SummaryWriter

import utils
import utils.train
import model


def load_mapper(path):
    with open(os.path.splitext(path)[0] + '.tsv', 'r') as f:
        lines = list(csv.reader(f, delimiter='\t'))
    mapper = {}
    for line in lines:
        if line:
            if len(line) < 3:
                line += [''] * (3 - len(line))
            dst, src, _converter = line
            converter = eval(_converter) if _converter else lambda val: val
            mapper[dst] = (src, converter)
    return mapper


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    torch.manual_seed(args.seed)
    mapper = load_mapper(os.path.expandvars(os.path.expanduser(args.mapper)))
    model_dir = utils.get_model_dir(config)
    _, num_parts = utils.get_dataset_mappers(config)
    limbs_index = utils.get_limbs_index(config)
    height, width = tuple(map(int, config.get('image', 'size').split()))
    tensor = torch.randn(args.batch_size, 3, height, width)
    # PyTorch
    try:
        path, step, epoch = utils.train.load_model(model_dir)
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    except (FileNotFoundError, ValueError):
        state_dict = {name: None for name in ('dnn', 'stages')}
    config_channels_dnn = model.ConfigChannels(config, state_dict['dnn'])
    dnn = utils.parse_attr(config.get('model', 'dnn'))(config_channels_dnn)
    config_channels_stages = model.ConfigChannels(config, state_dict['stages'], config_channels_dnn.channels)
    channel_dict = model.channel_dict(num_parts, len(limbs_index))
    stages = nn.Sequential(*[utils.parse_attr(s)(config_channels_stages, channel_dict, config_channels_dnn.channels, str(i)) for i, s in enumerate(config.get('model', 'stages').split())])
    inference = model.Inference(config, dnn, stages)
    inference.eval()
    state_dict = inference.state_dict()
    # TensorFlow
    with open(os.path.expanduser(os.path.expandvars(args.path)), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    image = ops.convert_to_tensor(np.transpose(tensor.cpu().numpy(), [0, 2, 3, 1]), name='image')
    tf.import_graph_def(graph_def, input_map={'image:0': image})
    saver = utils.train.Saver(model_dir, config.getint('save', 'keep'))
    with tf.Session(config=tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )) as sess:
        try:
            for dst in state_dict:
                src, converter = mapper[dst]
                if src.isdigit():
                    state_dict[dst].fill_(float(src))
                else:
                    op = sess.graph.get_operation_by_name(src)
                    t = op.values()[0]
                    v = sess.run(t)
                    state_dict[dst] = torch.from_numpy(converter(v))
                val = state_dict[dst].numpy()
                print('\t'.join(list(map(str, (dst, src, val.shape, utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest())))))
            inference.load_state_dict(state_dict)
            if args.delete:
                logging.warning('delete model directory: ' + model_dir)
                shutil.rmtree(model_dir, ignore_errors=True)
            saver(dict(
                dnn=inference.dnn.state_dict(),
                stages=inference.stages.state_dict(),
            ), 0)
        finally:
            if args.debug:
                for op in sess.graph.get_operations():
                    if op.values():
                        logging.info(op.values()[0])
                for name in args.debug:
                    t = sess.graph.get_tensor_by_name(name + ':0')
                    val = sess.run(t)
                    val = np.transpose(val, [0, 3, 1, 2])
                    print('\t'.join(map(str, [
                        name,
                        'x'.join(map(str, val.shape)),
                        utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest(),
                    ])))
            val = dnn(tensor).detach().numpy()
            print('\t'.join(map(str, [
                'x'.join(map(str, val.shape)),
                utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest(),
            ])))
            for stage, output in enumerate(inference(tensor)):
                for name, feature in output.items():
                    val = feature.detach().numpy()
                    print('\t'.join(map(str, [
                        'stage%d/%s' % (stage, name),
                        'x'.join(map(str, val.shape)),
                        utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest(),
                    ])))
            forward = inference.forward
            inference.forward = lambda self, *x: list(forward(self, *x)[-1].values())
            with SummaryWriter(model_dir) as writer:
                writer.add_graph(inference, (tensor,))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('mapper')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-d', '--delete', action='store_true', help='delete model')
    parser.add_argument('-s', '--seed', default=0, type=int, help='a seed to create a random image tensor')
    parser.add_argument('--debug', nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    main()