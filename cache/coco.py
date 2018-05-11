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
import logging
import configparser

import numpy as np
import pandas as pd
import tqdm
import pycocotools.coco
import pycocotools.mask
from PIL import Image, ImageDraw

import utils
import utils.cache


def draw_mask(segmentation, canvas, draw):
    pixels = canvas.load()
    if isinstance(segmentation, list):
        for polygon in segmentation:
            draw.polygon(polygon, fill=0)
    else:
        if isinstance(segmentation['counts'], list):
            rle = pycocotools.mask.frPyObjects([segmentation], canvas.size[1], canvas.size[0])
        else:
            rle = [segmentation]
        m = np.squeeze(pycocotools.mask.decode(rle))
        assert m.shape[:2] == canvas.size[::-1]
        for y, row in enumerate(m):
            for x, v in enumerate(row):
                if v:
                    pixels[x, y] = 0


def cache(config, path, mapper):
    name = __name__.split('.')[-1]
    cachedir = os.path.dirname(path)
    phase = os.path.splitext(os.path.basename(path))[0]
    phasedir = os.path.join(cachedir, phase)
    os.makedirs(phasedir, exist_ok=True)
    mask_ext = config.get('cache', 'mask_ext')
    data = []
    for i, row in pd.read_csv(os.path.splitext(__file__)[0] + '.tsv', sep='\t').iterrows():
        logging.info('loading data %d (%s)' % (i, ', '.join([k + '=' + str(v) for k, v in row.items()])))
        root = os.path.expanduser(os.path.expandvars(row['root']))
        year = str(row['year'])
        suffix = phase + year
        path = os.path.join(root, 'annotations', 'person_keypoints_%s.json' % suffix)
        if not os.path.exists(path):
            logging.warning(path + ' not exists')
            continue
        coco_kp = pycocotools.coco.COCO(path)
        skeleton = np.array(coco_kp.loadCats(1)[0]['skeleton']) - 1
        np.savetxt(os.path.join(os.path.dirname(cachedir), name + '.tsv'), skeleton, fmt='%d', delimiter='\t')
        imgIds = coco_kp.getImgIds()
        folder = os.path.join(root, suffix)
        imgs = coco_kp.loadImgs(imgIds)
        _imgs = list(filter(lambda img: os.path.exists(os.path.join(folder, img['file_name'])), imgs))
        if len(imgs) > len(_imgs):
            logging.warning('%d of %d images not exists' % (len(imgs) - len(_imgs), len(imgs)))
        for img in tqdm.tqdm(_imgs):
            # image
            path = os.path.join(folder, img['file_name'])
            width, height = img['width'], img['height']
            try:
                if config.getboolean('cache', 'verify'):
                    if not np.all(np.equal(utils.image_size(path), [width, height])):
                        logging.error('failed to verify shape of image ' + path)
                        continue
            except configparser.NoOptionError:
                pass
            # keypoints
            annIds = coco_kp.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco_kp.loadAnns(annIds)
            keypoints = []
            bbox = []
            keypath = os.path.join(phasedir, __name__.split('.')[-1] + year, os.path.relpath(os.path.splitext(path)[0], root))
            os.makedirs(os.path.dirname(keypath), exist_ok=True)
            maskpath = keypath + '.mask' + mask_ext
            with Image.new('L', (width, height), 255) as canvas:
                draw = ImageDraw.Draw(canvas)
                for ann in anns:
                    points = mapper(np.array(ann['keypoints']).reshape([-1, 3]))
                    if np.any(points[:, 2] > 0):
                        keypoints.append(points)
                        bbox.append(ann['bbox'])
                    else:
                        draw_mask(ann['segmentation'], canvas, draw)
                if len(keypoints) <= 0:
                    continue
                canvas.save(os.path.join(cachedir, maskpath))
            keypoints = np.array(keypoints, dtype=np.float32)
            keypoints = keypoints[:, :, [1, 0, 2]]
            bbox = np.array(bbox, dtype=np.float32)
            yx_min = bbox[:, 1::-1]
            size = bbox[:, -1:1:-1]
            yx_max = yx_min + size
            try:
                if config.getboolean('cache', 'dump'):
                    np.save(keypath + '.keypoints.npy', keypoints)
                    np.save(keypath + '.yx_min.npy', yx_min)
                    np.save(keypath + '.yx_max.npy', yx_max)
            except configparser.NoOptionError:
                pass
            data.append(dict(
                path=path, keypath=keypath,
                keypoints=keypoints,
                yx_min=yx_min, yx_max=yx_max,
            ))
        logging.warning('%d of %d images are saved' % (len(data), len(_imgs)))
    return data
