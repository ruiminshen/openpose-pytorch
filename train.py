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
import argparse
import configparser
import logging
import logging.config
import collections
import multiprocessing
import os
import shutil
import io
import hashlib
import subprocess
import pickle
import traceback
import yaml

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms
import tqdm
import humanize
import pybenchmark
import filelock
from tensorboardX import SummaryWriter
import cv2
import pyopenpose

import model
import transform.augmentation
import utils.data
#import utils.postprocess
import utils.train
import utils.visualize
#import eval as _eval


def ensure_model(model):
    if torch.cuda.is_available():
        model.cuda()
        if torch.cuda.device_count() > 1:
            logging.info('%d GPUs are used' % torch.cuda.device_count())
            model = nn.DataParallel(model).cuda()
    return model


class SummaryWorker(multiprocessing.Process):
    def __init__(self, env):
        super(SummaryWorker, self).__init__()
        self.env = env
        self.config = env.config
        self.queue = multiprocessing.Queue()
        try:
            self.timer_scalar = utils.train.Timer(env.config.getfloat('summary', 'scalar'))
        except configparser.NoOptionError:
            self.timer_scalar = lambda: False
        try:
            self.timer_image = utils.train.Timer(env.config.getfloat('summary', 'image'))
        except configparser.NoOptionError:
            self.timer_image = lambda: False
        try:
            self.timer_histogram = utils.train.Timer(env.config.getfloat('summary', 'histogram'))
        except configparser.NoOptionError:
            self.timer_histogram = lambda: False
        with open(os.path.expanduser(os.path.expandvars(env.config.get('summary_histogram', 'parameters'))), 'r') as f:
            self.histogram_parameters = utils.RegexList([line.rstrip() for line in f])
        self.draw_points = utils.visualize.DrawPoints(env.limbs_index, colors=env.config.get('draw_points', 'colors').split())
        self._draw_points = utils.visualize.DrawPoints(env.limbs_index, thickness=1)
        self.draw_bbox = utils.visualize.DrawBBox()
        self.draw_feature = utils.visualize.DrawFeature()
        self.draw_cluster = utils.visualize.DrawCluster()

    def __call__(self, name, **kwargs):
        if getattr(self, 'timer_' + name)():
            kwargs = getattr(self, 'copy_' + name)(**kwargs)
            self.queue.put((name, kwargs))

    def stop(self):
        self.queue.put((None, {}))

    def run(self):
        self.writer = SummaryWriter(os.path.join(self.env.model_dir, self.env.args.run))
        try:
            height, width = tuple(map(int, self.config.get('image', 'size').split()))
            tensor = torch.randn(1, 3, height, width)
            step, epoch, dnn, stages = self.env.load()
            inference = model.Inference(self.config, dnn, stages)
            forward = inference.forward
            inference.forward = lambda self, *x: list(forward(self, *x)[-1].values())
            self.writer.add_graph(inference, (torch.autograd.Variable(tensor),))
        except:
            traceback.print_exc()
        while True:
            name, kwargs = self.queue.get()
            if name is None:
                break
            func = getattr(self, 'summary_' + name)
            try:
                func(**kwargs)
            except:
                traceback.print_exc()

    def copy_scalar(self, **kwargs):
        step, loss_total, losses, losses_hparam = (kwargs[key] for key in 'step, loss_total, losses, losses_hparam'.split(', '))
        loss_total = loss_total.data.clone().cpu().numpy()
        losses = [{name: l.data.clone().cpu().numpy() for name, l in loss.items()} for loss in losses]
        losses_hparam = [{name: l.data.clone().cpu().numpy() for name, l in loss.items()} for loss in losses_hparam]
        return dict(
            step=step,
            loss_total=loss_total,
            losses=losses, losses_hparam=losses_hparam,
        )

    def summary_scalar(self, **kwargs):
        step, loss_total, losses, losses_hparam = (kwargs[key] for key in 'step, loss_total, losses, losses_hparam'.split(', '))
        for i, loss in enumerate(losses):
            for name, l in loss.items():
                self.writer.add_scalar('loss/%s%d' % (name, i), l, step)
        if self.config.getboolean('summary_scalar', 'loss_hparam'):
            self.writer.add_scalars('loss_hparam', {'%s%d' % (name, i): l for name, l in loss.items() for i, loss in enumerate(losses_hparam)}, step)
        self.writer.add_scalar('loss_total', loss_total, step)

    def copy_image(self, **kwargs):
        step, height, width, data, outputs = (kwargs[key] for key in 'step, height, width, data, outputs'.split(', '))
        image, mask, keypoints, yx_min, yx_max, parts, limbs, index = (data[key].clone().cpu().numpy() for key in 'image, mask, keypoints, yx_min, yx_max, parts, limbs, index'.split(', '))
        output = outputs[self.config.getint('summary_image', 'stage')]
        output = {name: output[name].data.clone().cpu().numpy() for name in self.config.get('summary_image', 'output').split()}
        return dict(
            step=step, height=height, width=width,
            image=image, mask=mask, keypoints=keypoints, yx_min=yx_min, yx_max=yx_max, parts=parts, limbs=limbs, index=index,
            output=output,
        )

    def summary_image(self, **kwargs):
        step, height, width, image, mask, keypoints, yx_min, yx_max, parts, limbs, index, output = (kwargs[key] for key in 'step, height, width, image, mask, keypoints, yx_min, yx_max, parts, limbs, index, output'.split(', '))
        limit = min(self.config.getint('summary_image', 'limit'), image.shape[0])
        image = image[:limit, :, :, :]
        if self.config.getboolean('summary_image', 'estimate'):
            canvas = np.copy(image)
            fn = pybenchmark.profile('output/estimate')(self.draw_clusters)
            canvas = [fn(canvas, parts[:-1], limbs) for canvas, parts, limbs in zip(canvas, *(output[name] for name in 'parts, limbs'.split(', ')))]
            self.writer.add_image('output/estimate', torchvision.utils.make_grid(torch.from_numpy(np.stack(canvas)).permute(0, 3, 1, 2).float(), normalize=True, scale_each=True), step)
        if self.config.getboolean('summary_image', 'data_keypoints'):
            canvas = np.copy(image)
            fn = pybenchmark.profile('data/keypoints')(self.draw_keypoints)
            canvas = [fn(canvas, mask, keypoints, yx_min, yx_max, index) for canvas, mask, keypoints, yx_min, yx_max, index in zip(canvas, mask, keypoints, yx_min, yx_max, index)]
            self.writer.add_image('data/keypoints', torchvision.utils.make_grid(torch.from_numpy(np.stack(canvas)).permute(0, 3, 1, 2).float(), normalize=True, scale_each=True), step)
        if self.config.getboolean('summary_image', 'data_parts'):
            fn = pybenchmark.profile('data/parts')(self.draw_feature)
            for i in range(parts.shape[1]):
                canvas = np.copy(image)
                canvas = [fn(canvas, feature[i]) for canvas, feature in zip(canvas, parts)]
                self.writer.add_image('data/parts%d' % i, torchvision.utils.make_grid(torch.from_numpy(np.stack(canvas)).permute(0, 3, 1, 2).float(), normalize=True, scale_each=True), step)
        if self.config.getboolean('summary_image', 'data_limbs'):
            fn = pybenchmark.profile('data/limbs')(self.draw_feature)
            for i in range(limbs.shape[1]):
                canvas = np.copy(image)
                canvas = [fn(canvas, feature[i]) for canvas, feature in zip(canvas, limbs)]
                self.writer.add_image('data/limbs%d' % i, torchvision.utils.make_grid(torch.from_numpy(np.stack(canvas)).permute(0, 3, 1, 2).float(), normalize=True, scale_each=True), step)
        for name, feature in output.items():
            fn = pybenchmark.profile('output/' + name)(self.draw_feature)
            for i in range(feature.shape[1]):
                canvas = np.copy(image)
                canvas = [fn(canvas, feature[i]) for canvas, feature in zip(canvas, feature)]
                self.writer.add_image('output/%s%d' % (name, i), torchvision.utils.make_grid(torch.from_numpy(np.stack(canvas)).permute(0, 3, 1, 2).float(), normalize=True, scale_each=True), step)

    def draw_keypoints(self, image, mask, keypoints, yx_min, yx_max, index):
        image = utils.visualize.draw_mask(image, mask, 1)
        size = yx_max - yx_min
        target = np.logical_and(*[np.squeeze(a, -1) > 0 for a in np.split(size, size.shape[-1], -1)])
        keypoints, yx_min, yx_max = (a[target] for a in (keypoints, yx_min, yx_max))
        for i, points in enumerate(keypoints):
            if i == index:
                image = self.draw_points(image, points)
            else:
                image = self._draw_points(image, points)
        image = self.draw_bbox(image, yx_min.astype(np.int), yx_max.astype(np.int))
        return image

    def draw_clusters(self, image, parts, limbs):
        try:
            interpolation = getattr(cv2, 'INTER_' + self.config.get('estimate', 'interpolation').upper())
            parts, limbs = (np.stack([cv2.resize(feature, image.shape[1::-1], interpolation=interpolation) for feature in a]) for a in (parts, limbs))
        except configparser.NoOptionError:
            pass
        clusters = pyopenpose.estimate(
            parts, limbs,
            self.env.limbs_index,
            self.config.getfloat('nms', 'threshold'),
            self.config.getfloat('integration', 'step'), tuple(map(int, self.config.get('integration', 'step_limits').split())), self.config.getfloat('integration', 'min_score'), self.config.getint('integration', 'min_count'),
            self.config.getfloat('cluster', 'min_score'), self.config.getint('cluster', 'min_count'),
        )
        scale_y, scale_x = np.array(image.shape[1::-1], parts.dtype) / np.array(parts.shape[-2:], parts.dtype)
        for cluster in clusters:
            cluster = [((i1, int(y1 * scale_y), int(x1 * scale_x)), (i2, int(y2 * scale_y), int(x2 * scale_x))) for (i1, y1, x1), (i2, y2, x2) in cluster]
            image = self.draw_cluster(image, cluster)
        return image

    def copy_histogram(self, **kwargs):
        return {
            'step': kwargs['step'],
            'state_dict': self.env.dnn.state_dict(),
        }

    def summary_histogram(self, **kwargs):
        step, state_dict = (kwargs[key] for key in 'step, state_dict'.split(', '))
        for name, var in state_dict.items():
            if self.histogram_parameters(name):
                self.writer.add_histogram(name, var, step)


class Train(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.model_dir = utils.get_model_dir(config)
        self.cache_dir = utils.get_cache_dir(config)
        _, self.num_parts = utils.get_dataset_mappers(config)
        self.limbs_index = utils.get_limbs_index(config)
        logging.info('use cache directory ' + self.cache_dir)
        logging.info('tensorboard --logdir ' + self.model_dir)
        if args.delete:
            logging.warning('delete model directory: ' + self.model_dir)
            shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_dir + '.ini', 'w') as f:
            config.write(f)

        self.step, self.epoch, self.dnn, self.stages = self.load()
        self.inference = model.Inference(self.config, self.dnn, self.stages)
        logging.info(humanize.naturalsize(sum(var.cpu().numpy().nbytes for var in self.inference.state_dict().values())))
        if self.args.finetune:
            path = os.path.expanduser(os.path.expandvars(self.args.finetune))
            logging.info('finetune from ' + path)
            self.finetune(self.dnn, path)
        self.inference = ensure_model(self.inference)
        self.inference.train()
        self.optimizer = eval(self.config.get('train', 'optimizer'))(filter(lambda p: p.requires_grad, self.inference.parameters()), self.args.learning_rate)

        self.saver = utils.train.Saver(self.model_dir, config.getint('save', 'keep'))
        self.timer_save = utils.train.Timer(config.getfloat('save', 'secs'), False)
        try:
            self.timer_eval = utils.train.Timer(eval(config.get('eval', 'secs')), config.getboolean('eval', 'first'))
        except configparser.NoOptionError:
            self.timer_eval = lambda: False
        self.summary_worker = SummaryWorker(self)
        self.summary_worker.start()

    def stop(self):
        self.summary_worker.stop()
        self.summary_worker.join()

    def get_loader(self, dnn):
        paths = [os.path.join(self.cache_dir, phase + '.pkl') for phase in self.config.get('train', 'phase').split()]
        dataset = utils.data.Dataset(
            self.config,
            utils.data.load_pickles(paths),
            transform=transform.augmentation.get_transform(self.config, self.config.get('transform', 'augmentation').split()),
            shuffle=self.config.getboolean('data', 'shuffle'),
            dir=os.path.join(self.model_dir, 'exception'),
        )
        logging.info('num_examples=%d' % len(dataset))
        try:
            workers = self.config.getint('data', 'workers')
            if torch.cuda.is_available():
                workers = workers * torch.cuda.device_count()
        except configparser.NoOptionError:
            workers = multiprocessing.cpu_count()
        sizes = utils.train.load_sizes(self.config)
        feature_sizes = [model.feature_size(dnn, *size) for size in sizes]
        collate_fn = utils.data.Collate(
            self.config,
            transform.parse_transform(self.config, self.config.get('transform', 'resize_train')),
            sizes, feature_sizes,
            maintain=self.config.getint('data', 'maintain'),
            transform_image=transform.get_transform(self.config, self.config.get('transform', 'image_train').split()),
            transform_tensor=transform.get_transform(self.config, self.config.get('transform', 'tensor').split()),
            dir=os.path.join(self.model_dir, 'exception'),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size * torch.cuda.device_count() if torch.cuda.is_available() else self.args.batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    def load(self):
        try:
            path, step, epoch = utils.train.load_model(self.model_dir)
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        except (FileNotFoundError, ValueError):
            step, epoch = 0, 0
            state_dict = {name: None for name in ('dnn', 'stages')}
        config_channels_dnn = model.ConfigChannels(self.config, state_dict['dnn'])
        dnn = utils.parse_attr(self.config.get('model', 'dnn'))(config_channels_dnn)
        config_channels_stages = model.ConfigChannels(self.config, state_dict['stages'], config_channels_dnn.channels)
        channel_dict = model.channel_dict(self.num_parts, len(self.limbs_index))
        stages = nn.Sequential(*[utils.parse_attr(s)(config_channels_stages, channel_dict, config_channels_dnn.channels, str(i)) for i, s in enumerate(self.config.get('model', 'stages').split())])
        if config_channels_dnn.state_dict is not None:
            dnn.load_state_dict(config_channels_dnn.state_dict)
        if config_channels_stages.state_dict is not None:
            stages.load_state_dict(config_channels_stages.state_dict)
        return step, epoch, dnn, stages

    def finetune(self, model, path):
        if os.path.isdir(path):
            path, _step, _epoch = utils.train.load_model(path)
        _state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = model.state_dict()
        ignore = utils.RegexList(self.args.ignore)
        for key, value in state_dict.items():
            try:
                if not ignore(key):
                    state_dict[key] = _state_dict[key]
            except KeyError:
                logging.warning('%s not in finetune file %s' % (key, path))
        model.load_state_dict(state_dict)

    def loss_hparam(self, i, name, loss):
        try:
            return loss * self.config.getfloat('hparam', '%s%d' % (name, i))
        except configparser.NoOptionError:
            return loss * self.config.getfloat('hparam', name)

    def iterate(self, data):
        for key in data:
            t = data[key]
            if torch.is_tensor(t):
                data[key] = utils.ensure_device(t)
        tensor = torch.autograd.Variable(data['tensor'])
        outputs = pybenchmark.profile('inference')(self.inference)(tensor)
        height, width = data['image'].size()[1:3]
        loss = pybenchmark.profile('loss')(model.Loss(self.config, data, self.limbs_index, height, width))
        losses = [loss(**output) for output in outputs]
        losses_hparam = [{name: self.loss_hparam(i, name, l) for name, l in loss.items()} for i, loss in enumerate(losses)]
        loss_total = sum(sum(loss.values()) for loss in losses_hparam)
        self.optimizer.zero_grad()
        loss_total.backward()
        try:
            clip = self.config.getfloat('train', 'clip')
            nn.utils.clip_grad_norm(self.inference.parameters(), clip)
        except configparser.NoOptionError:
            pass
        self.optimizer.step()
        return dict(
            height=height, width=width,
            data=data, outputs=outputs,
            loss_total=loss_total, losses=losses, losses_hparam=losses_hparam,
        )

    def __call__(self):
        with filelock.FileLock(os.path.join(self.model_dir, 'lock'), 0):
            try:
                try:
                    scheduler = eval(self.config.get('train', 'scheduler'))(self.optimizer)
                except configparser.NoOptionError:
                    scheduler = None
                loader = self.get_loader(self.dnn)
                logging.info('num_workers=%d' % loader.num_workers)
                step = self.step
                for epoch in range(0 if self.epoch is None else self.epoch, self.args.epoch):
                    if scheduler is not None:
                        scheduler.step(epoch)
                        logging.info('epoch=%d, lr=%s' % (epoch, str(scheduler.get_lr())))
                    for data in loader if self.args.quiet else tqdm.tqdm(loader, desc='epoch=%d/%d' % (epoch, self.args.epoch)):
                        kwargs = self.iterate(data)
                        step += 1
                        kwargs = {**kwargs, **dict(step=step, epoch=epoch)}
                        self.summary_worker('scalar', **kwargs)
                        self.summary_worker('image', **kwargs)
                        self.summary_worker('histogram', **kwargs)
                        if self.timer_save():
                            self.save(**kwargs)
                        if self.timer_eval():
                            self.eval(**kwargs)
                self.save(**kwargs)
                logging.info('finished')
            except KeyboardInterrupt:
                logging.warning('interrupted')
                self.save(**kwargs)
            except:
                traceback.print_exc()
                try:
                    with open(os.path.join(self.model_dir, 'data.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                except UnboundLocalError:
                    pass
                raise
            finally:
                self.stop()

    def check_nan(self, **kwargs):
        step, loss_total, losses, data = (kwargs[key] for key in 'step, loss_total, losses, data'.split(', '))
        if np.isnan(loss_total.data.cpu()[0]):
            dump_dir = os.path.join(self.model_dir, str(step))
            os.makedirs(dump_dir, exist_ok=True)
            torch.save({name: collections.OrderedDict([(key, var.cpu()) for key, var in getattr(self, name).state_dict().items()]) for name in 'dnn, stages'.split(', ')}, os.path.join(dump_dir, 'model.pth'))
            torch.save(data, os.path.join(dump_dir, 'data.pth'))
            for i, loss in enumerate(losses):
                for name, l in loss.items():
                    logging.warning('%s%d=%f' % (name, i, l.data.cpu()[0]))
            raise OverflowError('NaN loss detected, dump runtime information into ' + dump_dir)

    def save(self, **kwargs):
        step, epoch = (kwargs[key] for key in 'step, epoch'.split(', '))
        self.check_nan(**kwargs)
        self.saver({name: collections.OrderedDict([(key, var.cpu()) for key, var in getattr(self, name).state_dict().items()]) for name in 'dnn, stages'.split(', ')}, step, epoch)

    def eval(self, **kwargs):
        logging.info('evaluating')
        if torch.cuda.is_available():
            self.inference.cpu()
        try:
            e = _eval.Eval(self.args, self.config)
            cls_ap = e()
            self.backup_best(cls_ap, e.path)
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            self.inference.cuda()

    def backup_best(self, cls_ap, path):
        try:
            with open(self.model_dir + '.pkl', 'rb') as f:
                best = np.mean(list(pickle.load(f).values()))
        except:
            best = np.finfo(np.float32).min
        metric = np.mean(list(cls_ap.values()))
        if metric > best:
            with open(self.model_dir + '.pkl', 'wb') as f:
                pickle.dump(cls_ap, f)
            shutil.copy(path, self.model_dir + '.pth')
            logging.info('best model (%f) saved into %s.*' % (metric, self.model_dir))
        else:
            logging.info('best metric %f >= %f' % (best, metric))


def main():
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    for cmd in args.modify:
        utils.modify_config(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    if args.run is None:
        buffer = io.StringIO()
        config.write(buffer)
        args.run = hashlib.md5(buffer.getvalue().encode()).hexdigest()
    logging.info('cd ' + os.getcwd() + ' && ' + subprocess.list2cmdline([sys.executable] + sys.argv))
    train = Train(args, config)
    train()
    logging.info(pybenchmark.stats)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', default=[], help='modify config')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('-f', '--finetune')
    parser.add_argument('-i', '--ignore', nargs='+', default=[], help='regex to ignore weights while fintuning')
    parser.add_argument('-lr', '--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('-e', '--epoch', type=int, default=np.iinfo(np.int).max)
    parser.add_argument('-d', '--delete', action='store_true', help='delete model')
    parser.add_argument('-q', '--quiet', action='store_true', help='quiet mode')
    parser.add_argument('-r', '--run', help='the run name in TensorBoard')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    return parser.parse_args()


if __name__ == '__main__':
    main()
