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
import caffe

import utils
import utils.train
import model


def load_mapper(path):
    with open(path, 'r') as f:
        lines = list(csv.reader(f, delimiter='\t'))
    mapper = {}
    for line in lines:
        if len(line) == 3:
            dst, src, transform = line
            transform = eval(transform)
            mapper[dst] = (src, transform)
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
    # Caffe
    net = caffe.Net(model_dir + '.prototxt', model_dir + '.caffemodel', caffe.TEST)
    if args.debug:
        logging.info('Caffe variables')
        for name, blobs in net.params.items():
            for i, blob in enumerate(blobs):
                val = blob.data
                print('\t'.join(map(str, [
                    '%s/%d' % (name, i),
                    'x'.join(map(str, val.shape)),
                    utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest(),
                ])))
        logging.info('Caffe features')
        input = net.blobs[args.input]
        input.reshape(*tensor.size())
        input.data[...] = tensor.numpy()
        net.forward()
        for name, blob in net.blobs.items():
            val = blob.data
            print('\t'.join(map(str, [
                name,
                'x'.join(map(str, val.shape)),
                utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest(),
            ])))
    # convert
    saver = utils.train.Saver(model_dir, config.getint('save', 'keep'))
    try:
        for dst in state_dict:
            src, transform = mapper[dst]
            blobs = [b.data for b in net.params[src]]
            blob = transform(blobs)
            if isinstance(blob, np.ndarray):
                state_dict[dst] = torch.from_numpy(blob)
            else:
                state_dict[dst].fill_(blob)
            val = state_dict[dst].numpy()
            logging.info('\t'.join(list(map(str, (dst, src, val.shape, utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest())))))
        inference.load_state_dict(state_dict)
        if args.delete:
            logging.warning('delete model directory: ' + model_dir)
            shutil.rmtree(model_dir, ignore_errors=True)
        saver(dict(
            dnn=inference.dnn.state_dict(),
            stages=inference.stages.state_dict(),
        ), 0)
    finally:
        for stage, output in enumerate(inference(torch.autograd.Variable(tensor, volatile=True))):
            for name, feature in output.items():
                val = feature.data.numpy()
                print('\t'.join(map(str, [
                    'stage%d/%s' % (stage, name),
                    'x'.join(map(str, val.shape)),
                    utils.abs_mean(val), hashlib.md5(val.tostring()).hexdigest(),
                ])))


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mapper')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-i', '--input', default='image', help='input tensor name of Caffe')
    parser.add_argument('-d', '--delete', action='store_true', help='delete model')
    parser.add_argument('-s', '--seed', default=0, type=int, help='a seed to create a random image tensor')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main()
