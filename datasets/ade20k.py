"""
ADE20K Dataset Loader
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import logging
import datasets.uniform as uniform
import datasets.edge_utils as edge_utils
import json
from config import cfg

num_classes = 150
ignore_label = 255
root = cfg.DATASET.ADE20K_DIR

palette = [120, 120, 120,
           180, 120, 120,
            6, 230, 230,
            80, 50, 50,
            4, 200, 3,
            120, 120, 80,
            140, 140, 140,
            204, 5, 255,
            230, 230, 230,
            4, 250, 7,
            224, 5, 255,
            235, 255, 7,
            150, 5, 61,
            120, 120, 70,
            8, 255, 51,
            255, 6, 82,
            143, 255, 140,
            204, 255, 4,
            255, 51, 7,
            204, 70, 3,
            0, 102, 200,
            61, 230, 250,
            255, 6, 51,
            11, 102, 255,
            255, 7, 71,
            255, 9, 224,
            9, 7, 230,
            220, 220, 220,
            255, 9, 92,
            112, 9, 255,
            8, 255, 214,
            7, 255, 224,
            255, 184, 6,
            10, 255, 71,
            255, 41, 10,
            7, 255, 255,
            224, 255, 8,
            102, 8, 255,
            255, 61, 6,
            255, 194, 7,
            255, 122, 8,
            0, 255, 20,
            255, 8, 41,
            255, 5, 153,
            6, 51, 255,
            235, 12, 255,
            160, 150, 20,
            0, 163, 255,
            140, 140, 140,
            250, 10, 15,
            20, 255, 0,
            31, 255, 0,
            255, 31, 0,
            255, 224, 0,
            153, 255, 0,
            0, 0, 255,
            255, 71, 0,
            0, 235, 255,
            0, 173, 255,
            31, 0, 255,
            11, 200, 200,
            255 ,82, 0,
            0, 255, 245,
            0, 61, 255,
            0, 255, 112,
            0, 255, 133,
            255, 0, 0,
            255, 163, 0,
            255, 102, 0,
            194, 255, 0,
            0, 143, 255,
            51, 255, 0,
            0, 82, 255,
            0, 255, 41,
            0, 255, 173,
            10, 0, 255,
            173, 255, 0,
            0, 255, 153,
            255, 92, 0,
            255, 0, 255,
            255, 0, 245,
            255, 0, 102,
            255, 173, 0,
            255, 0, 20,
            255, 184, 184,
            0, 31, 255,
            0, 255, 61,
            0, 71, 255,
            255, 0, 204,
            0, 255, 194,
            0, 255, 82,
            0, 10, 255,
            0, 112, 255,
            51, 0, 255,
            0, 194, 255,
            0, 122, 255,
            0, 255, 163,
            255, 153, 0,
            0, 255, 10,
            255, 112, 0,
            143, 255, 0,
            82, 0, 255,
            163, 255, 0,
            255, 235, 0,
            8, 184, 170,
            133, 0, 255,
            0, 255, 92,
            184, 0, 255,
            255, 0, 31,
            0, 184, 255,
            0, 214, 255,
            255, 0, 112,
            92, 255, 0,
            0, 224, 255,
            112, 224, 255,
            70, 184, 160,
            163, 0, 255,
            153, 0, 255,
            71, 255, 0,
            255, 0, 163,
            255, 204, 0,
            255, 0, 143,
            0, 255, 235,
            133, 255, 0,
            255, 0, 235,
            245, 0, 255,
            255, 0, 122,
            255, 245, 0,
            10, 190, 212,
            214, 255, 0,
            0, 204, 255,
            20, 0, 255,
            255, 255, 0,
            0, 153, 255,
            0, 41, 255,
            0, 255, 204,
            41, 0, 255,
            41, 255, 0,
            173, 0, 255,
            0, 245, 255,
            71, 0, 255,
            122, 0, 255,
            0, 255, 184,
            0, 92, 255,
            184, 255, 0,
            0, 133, 255,
            255, 214, 0,
            25, 194, 194,
            102, 255, 0,
            92, 0, 255,]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def add_items(img_path, mask_path, mode=None):
    c_items = os.listdir(img_path)
    c_items.sort()
    items = []
    aug_items = []
    if mode == 'test':
        for it in c_items:
            item = (os.path.join(img_path, it))
            items.append(item)
    else:
        for it in c_items:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0] + '.png'))
            items.append(item)
    return items, aug_items


def make_dataset(quality, mode):
    items = []
    aug_items = []
    assert quality == 'semantic'
    assert mode in ['train', 'val', 'trainval', 'test']

    img_path = os.path.join(root, 'train', 'image')
    mask_path = os.path.join(root, 'train', 'label')

    train_items, train_aug_items = add_items(img_path, mask_path)
    logging.info('ADE20K has a total of {} train images'.format(len(train_items)))

    img_path = os.path.join(root, 'val', 'image')
    mask_path = os.path.join(root, 'val', 'label')

    val_items, val_aug_items = add_items(img_path, mask_path)
    logging.info('ADE20K has a total of {} validation images'.format(len(val_items)))

    if mode == 'test':
        img_path = os.path.join(root, 'test', 'image')
        mask_path = os.path.join(root, 'test', 'label')
        test_items, test_aug_items = add_items(img_path, mask_path, mode=mode)
        logging.info('ADE20K has a total of {} test images'.format(len(test_items)))
        
    if mode == 'train':
        items = train_items
    elif mode == 'val':
        items = val_items
    elif mode == 'trainval':
        items = train_items + val_items
        aug_items = train_aug_items + val_aug_items
    elif mode == 'test':
        items = test_items
        aug_items = []
    else:
        logging.info('unknown mdoe {}'.format(mode))
        sys.exit()
    logging.info('ADE20K-{}: {} images'.format(mode, len(items)))
    
    return items, aug_items


class ADE20K(data.Dataset):
    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0, class_uniform_tile=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False):
        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.scf = scf
        self.hardnm = hardnm
        self.cv_split = cv_split
        self.edge_map = edge_map
        self.centroids = []

        self.imgs, self.aug_imgs = make_dataset(quality, mode)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for GT data
        if self.class_uniform_pct > 0:
            json_fn = 'bdd_tile{}_cv{}_{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode)

            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                self.centroids = uniform.class_centroids_all(
                    self.imgs,
                    num_classes,
                    id2trainid=None,
                    tile_size=class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)

            self.fine_centroids = self.centroids.copy()

            if self.maxSkip > 0:
                json_fn = 'bdd_tile{}_cv{}_{}_skip{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode,
                                                                  self.maxSkip)
                if os.path.isfile(json_fn):
                    with open(json_fn, 'r') as json_data:
                        centroids = json.load(json_data)
                    self.aug_centroids = {int(idx): centroids[idx] for idx in centroids}
                else:
                    self.aug_centroids = uniform.class_centroids_all(
                        self.aug_imgs,
                        num_classes,
                        id2trainid=None,
                        tile_size=class_uniform_tile)
                    with open(json_fn, 'w') as outfile:
                        json.dump(self.aug_centroids, outfile, indent=4)

                for class_id in range(num_classes):
                    self.centroids[class_id].extend(self.aug_centroids[class_id])

        self.build_epoch()

    def build_epoch(self, cut=False):

        if self.class_uniform_pct > 0:
            if cut:
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                self.fine_centroids,
                                                num_classes,
                                                cfg.CLASS_UNIFORM_PCT)
            else:
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                self.centroids,
                                                num_classes,
                                                cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            if self.mode == 'test':
                img_path = elem
                mask_path = None
            else:
                img_path, mask_path = elem

        if mask_path is not None:
            mask = Image.open(mask_path)
            # print('original', np.min(np.array(mask)), np.max(np.array(mask)))
            # Image to numpy
            mask = np.array(mask)
            mask = mask - 1
            mask[mask == -1] = 255
            mask = Image.fromarray(mask)
            # print('after', np.min(np.array(mask)), np.max(np.array(mask)))
            # mask = mask - 1
            # mask has to reduce zero
        else:
            mask = None

        img = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        # Debug
        if self.dump_images and centroid is not None:
            outdir = './dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            dump_img_name = trainid_to_name[class_id] + '_' + img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and mask is not None:
            mask = self.target_transform(mask)

        if self.edge_map:
            # _edgemap = np.array(mask_trained)
            # _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
            _edgemap = mask[:-1, :, :]
            _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
            edgemap = torch.from_numpy(_edgemap).float()
            return img, mask, edgemap, img_name
        if mask is None:
            return img, img_name
        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)
