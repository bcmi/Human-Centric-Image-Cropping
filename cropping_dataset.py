import os
import cv2
import numpy as np
import pickle
import lmdb
import datetime
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import math
import random
from config_GAICD import cfg

MOS_MEAN = 2.95
MOS_STD  = 0.8
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# debug_dir = './test_dataset'
# os.makedirs(debug_dir, exist_ok=True)

def rescale_bbox(bbox, ratio_w, ratio_h):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:, 0] = np.floor(bbox[:, 0] * ratio_w)
    bbox[:, 1] = np.floor(bbox[:, 1] * ratio_h)
    bbox[:, 2] = np.ceil(bbox[:, 2] * ratio_w)
    bbox[:, 3] = np.ceil(bbox[:, 3] * ratio_h)
    return bbox.astype(np.float32)

def generate_crop_mask(bbox, width, height, downsample):
    bbox = np.array(bbox).reshape(-1, 4)
    target_w, target_h = int(width / downsample), int(height / downsample)
    bbox[:,0::2] *= (float(target_w) / width)
    bbox[:,1::2] *= (float(target_h) / height)
    bbox = bbox.astype(np.int32)
    mask = np.zeros((bbox.shape[0], target_h, target_w))
    for i in range(bbox.shape[0]):
        x1,y1,x2,y2 = bbox[i]
        mask[i, y1:y2, x1:x2] = 1
    mask = mask.astype(np.float32)
    return mask

def generate_partition_mask(bbox, width, height, downsample):
    bbox = np.array(bbox).reshape(-1, 4)
    target_w, target_h = int(width / downsample), int(height / downsample)
    bbox[:, 0::2] *= (float(target_w) / width)
    bbox[:, 1::2] *= (float(target_h) / height)
    bbox = bbox.astype(np.int32)
    x1, y1, x2, y2 = bbox[0]
    hor = [0, x1, x2, target_w]
    ver = [0, y1, y2, target_h]
    mask = np.zeros((9, target_h, target_w))
    if x1 >= 0:
        if x2 - x1 == 0:
            x2 = x1 + 1
        if y2 - y1 == 0:
            y2 = y1 + 1
        for i in range(len(ver) - 1):
            for j in range(len(hor) - 1):
                ch = i * 3 + j
                mask[ch, ver[i]: ver[i + 1], hor[j]: hor[j + 1]] = 1
    mask = mask.astype(np.float32)
    return mask

def generate_target_size_crop_mask(bbox, current_w, current_h, target_w, target_h):
    bbox = np.array(bbox).reshape(-1, 4)
    bbox[:,0::2] *= (float(target_w) / current_w)
    bbox[:,1::2] *= (float(target_h) / current_h)
    bbox = bbox.astype(np.int32)
    mask = np.zeros((bbox.shape[0], target_h, target_w))
    for i in range(bbox.shape[0]):
        x1,y1,x2,y2 = bbox[i]
        if x1 >= 0:
            mask[i, y1:y2, x1:x2] = 1
    mask = mask.astype(np.float32)
    return mask

class CPCDataset(Dataset):
    def __init__(self, only_human_images=False,
                 keep_aspect_ratio=True):
        self.only_human = only_human_images
        self.image_dir = cfg.CPC_image
        self.heat_map_dir = cfg.CPC_heat_map
        self.heat_map_scale = cfg.heat_map_size
        self.keep_aspect = keep_aspect_ratio
        assert os.path.exists(self.image_dir), self.image_dir
        self.human_bboxes = json.load(open(cfg.CPC_human, 'r'))
        self.annotations  = json.load(open(cfg.CPC_anno,  'r'))
        self.score_mean, self.score_std = self.statistic_score()
        if self.only_human:
            self.image_list = list(self.human_bboxes.keys())
        else:
            self.image_list = list(self.annotations.keys())
        self.augmentation = cfg.data_augmentation
        self.PhotometricDistort = transforms.ColorJitter(
            brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])
        self.heat_map_transformer = transforms.ToTensor()
        self.crop_mask_downsample = 4
        self.human_mask_downsample = 16

    def statistic_crop(self):
        overlap_thresh = 0.1
        total_cnt = 0
        nonhuman_cnt = 0
        nonhuman_score = []
        for image_name in self.human_bboxes.keys():
            crop = self.annotations[image_name]['bboxes']
            score = self.annotations[image_name]['scores']
            hbox = self.human_bboxes[image_name]['bbox']
            crop = np.array(crop).reshape(-1, 4)
            hbox = np.array(hbox).reshape(-1, 4)
            score = np.array(score).reshape(-1)
            overlap = compute_overlap(crop, hbox)
            total_cnt += overlap.shape[0]
            nonhuman_cnt += (overlap < overlap_thresh).sum()
            nonhuman_score.extend(score[overlap < overlap_thresh].tolist())
        print('{} human images, {} crops, {} non-human, {:.2%}'.format(
            len(self.human_bboxes.keys()), total_cnt, nonhuman_cnt, float(nonhuman_cnt) / total_cnt
        ))
        nonhuman_score = np.array(nonhuman_score).reshape(-1)
        print('{} scores, mean={:.2f}, median={:.2f}, max={:.2f}, min={:.2f}'.format(
            nonhuman_score.shape[0], np.mean(nonhuman_score), np.median(nonhuman_score),
            np.max(nonhuman_score), np.min(nonhuman_score)
        ))

    def statistic_score(self):
        score_list = []
        for image_name in self.annotations.keys():
            score = self.annotations[image_name]['scores']
            score_list.extend(score)
        score = np.array(score_list).reshape(-1)
        mean = np.mean(score)
        std = np.std(score)
        print('CPC has {} score annotations, mean={:.2f}, std={:.2f}'.format(
            len(score), mean, std))
        return mean, std

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        assert os.path.exists(image_file), image_file
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width  * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]
        resized_image = image.resize((w,h), Image.ANTIALIAS)
        rs_width, rs_height = resized_image.size
        ratio_h = float(rs_height) / im_height
        ratio_w = float(rs_width) / im_width

        heat_map_file = os.path.join(self.heat_map_dir, os.path.splitext(image_name)[0] + '.png')
        assert os.path.exists(heat_map_file), heat_map_file
        heat_map = Image.open(heat_map_file)
        # hm_w, hm_h = int(rs_width * self.heat_map_scale), int(rs_height * self.heat_map_scale)
        hm_w = hm_h = cfg.image_size[0] // 4
        heat_map = heat_map.resize((hm_w, hm_h))
        crop = self.annotations[image_name]['bboxes']
        crop = rescale_bbox(crop, ratio_w, ratio_h)
        score = self.annotations[image_name]['scores']
        score = torch.tensor(score).reshape(-1).float()
        if image_name in self.human_bboxes:
            hbox = self.human_bboxes[image_name]['bbox']
            hbox = rescale_bbox(hbox, ratio_w, ratio_h)
        else:
            hbox = np.array([[-1, -1, -1, -1]]).astype(np.float32)

        if self.augmentation:
            if random.uniform(0,1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                heat_map = ImageOps.mirror(heat_map)
                temp_x1 = crop[:,0].copy()
                crop[:,0] = rs_width - crop[:,2]
                crop[:,2] = rs_width - temp_x1

                if image_name in self.human_bboxes:
                    # print('human mirror_before', hbox[0])
                    temp_x1 = hbox[:,0].copy()
                    hbox[:,0] = rs_width - hbox[:,2]
                    hbox[:,2] = rs_width - temp_x1
                    # print('human mirror after',  hbox)
            resized_image = self.PhotometricDistort(resized_image)
        # debug
        # if hbox[0,0] > 0:
        #     plt.subplot(2,2,1)
        #     plt.imshow(resized_image)
        #     plt.title('input image')
        #     plt.axis('off')
        #
        #     plt.subplot(2,2,2)
        #     x1,y1,x2,y2 = crop[0].astype(np.int32)
        #     best_crop = np.asarray(resized_image)[y1:y2,x1:x2]
        #     # print('best_crop', crop[0], best_crop.shape)
        #     plt.imshow(best_crop)
        #     plt.title('best crop')
        #     plt.axis('off')
        #
        #     plt.subplot(2, 2, 3)
        #     x1, y1, x2, y2 = crop[-1].astype(np.int32)
        #     worst_crop = np.asarray(resized_image)[y1:y2, x1:x2]
        #     # print('worst_crop', crop[-1], worst_crop.shape)
        #     plt.imshow(worst_crop)
        #     plt.title('worst crop')
        #     plt.axis('off')
        #
        #     if hbox[0,0] >= 0:
        #         plt.subplot(2, 2, 4)
        #         x1,y1,x2,y2 = hbox[0].astype(np.int32)
        #         human = np.asarray(resized_image)[y1:y2,x1:x2]
        #         # print('human bbox', hbox, human.shape)
        #         plt.imshow(human)
        #         plt.title('human')
        #         plt.axis('off')
        #     fig_file = os.path.join(debug_dir, os.path.splitext(image_name)[0] + '_crop.jpg')
        #     plt.savefig(fig_file)
        #     plt.close()
        im = self.image_transformer(resized_image)
        heat_map = self.heat_map_transformer(heat_map)
        # crop_mask = generate_crop_mask(crop, rs_width, rs_height,
        #                                self.crop_mask_downsample)
        crop_mask = generate_target_size_crop_mask(crop, rs_width, rs_height, 64, 64)
        partition_mask = generate_partition_mask(hbox, rs_width, rs_height,
                                                 self.human_mask_downsample)
        # debug
        # if hbox[0,0] > 0:
        #     plt.imshow(heat_map[0], cmap ='gray')
        #     plt.axis('off')
        #     fig_file = os.path.join(debug_dir, os.path.splitext(image_name)[0] + '_heat_map.jpg')
        #     plt.savefig(fig_file)
        #     plt.close()
        #
        #     for i in range(9):
        #         plt.subplot(3,3,i+1)
        #         plt.imshow(partition_mask[i], cmap ='gray')
        #         plt.axis('off')
        #     fig_file = os.path.join(debug_dir, os.path.splitext(image_name)[0] + '_partition_mask.jpg')
        #     plt.savefig(fig_file)
        #     plt.close()

        return im, crop, hbox, heat_map, crop_mask, partition_mask, score, im_width, im_height

class FCDBDataset(Dataset):
    def __init__(self, only_human_images=True,
                 keep_aspect_ratio=False):
        self.only_human = only_human_images
        self.keep_aspect = keep_aspect_ratio
        self.image_dir = cfg.FCDB_image
        assert os.path.exists(self.image_dir), self.image_dir
        self.human_bboxes = json.load(open(cfg.FCDB_human, 'r'))
        self.annotations = json.load(open(cfg.FCDB_anno, 'r'))
        if self.only_human:
            self.image_list = list(self.human_bboxes.keys())
        else:
            self.image_list = json.load(open(cfg.FCDB_split, 'r'))['test']
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])
        self.crop_mask_downsample = 4
        self.human_mask_downsample = 16

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        assert os.path.exists(image_file), image_file
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        im = self.image_transformer(resized_image)
        rs_width, rs_height = resized_image.size
        ratio_h = float(rs_height) / im_height
        ratio_w = float(rs_width) / im_width
        if image_name in self.human_bboxes:
            hbox = self.human_bboxes[image_name]
            hbox = rescale_bbox(hbox, ratio_w, ratio_h)
        else:
            hbox = np.array([[-1, -1, -1, -1]]).astype(np.float32)
        partition_mask = generate_partition_mask(hbox, rs_width, rs_height,
                                                 self.human_mask_downsample)

        crop = self.annotations[image_name]
        x,y,w,h = crop
        crop = torch.tensor([x,y,x+w,y+h])
        return im, crop, hbox, partition_mask, im_width, im_height

class FLMSDataset(Dataset):
    def __init__(self, only_human_images=True,
                 keep_aspect_ratio=False):
        self.only_human = only_human_images
        self.keep_aspect = keep_aspect_ratio
        self.image_dir = cfg.FLMS_image
        assert os.path.exists(self.image_dir), self.image_dir
        self.human_bboxes = json.load(open(cfg.FLMS_human, 'r'))
        self.annotations = json.load(open(cfg.FLMS_anno, 'r'))
        if self.only_human:
            self.image_list = list(self.human_bboxes.keys())
        else:
            self.image_list = list(self.annotations.keys())
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])
        self.crop_mask_downsample = 4
        self.human_mask_downsample = 16

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        assert os.path.exists(image_file), image_file
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        im = self.image_transformer(resized_image)
        rs_width, rs_height = resized_image.size
        ratio_h = float(rs_height) / im_height
        ratio_w = float(rs_width) / im_width
        if image_name in self.human_bboxes:
            hbox = self.human_bboxes[image_name]
            hbox = rescale_bbox(hbox, ratio_w, ratio_h)
        else:
            hbox = np.array([[-1, -1, -1, -1]]).astype(np.float32)
        partition_mask = generate_partition_mask(hbox, rs_width, rs_height,
                                                 self.human_mask_downsample)
        crop = self.annotations[image_name]
        keep_crop = []
        for box in crop:
            x1, y1, x2, y2 = box
            if (x2 > im_width or y2 > im_height):
                continue
            keep_crop.append(box)
        for i in range(10 - len(keep_crop)):
            keep_crop.append([-1, -1, -1, -1])
        crop = torch.tensor(keep_crop)
        return im, crop, hbox, partition_mask, im_width, im_height

class GAICDataset(Dataset):
    def __init__(self, only_human_images=False, split='all',
                 keep_aspect_ratio=True):
        self.only_human = only_human_images
        self.split = split
        self.keep_aspect = keep_aspect_ratio
        self.image_size = cfg.image_size
        self.image_dir = cfg.GAIC_image
        self.heat_map_dir = cfg.GAIC_heat_map
        assert os.path.exists(self.image_dir), self.image_dir
        self.human_bboxes = json.load(open(cfg.GAIC_human, 'r'))
        self.annotations  = json.load(open(cfg.GAIC_anno, 'r'))
        self.data_split   = json.load(open(cfg.GAIC_split, 'r'))
        if self.only_human:
            if self.split == 'all':
                self.image_list = list(self.human_bboxes.keys())
            else:
                self.image_list = json.load(open(cfg.GAIC_human_split, 'r'))[self.split]
        else:
            if self.split == 'all':
                self.image_list = self.data_split['test'] + self.data_split['train']
            else:
                self.image_list = self.data_split[split]
        self.augmentation = (cfg.data_augmentation and split == 'train')
        self.PhotometricDistort = transforms.ColorJitter(
            brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])
        self.heat_map_transformer = transforms.ToTensor()
        self.crop_mask_downsample = 4
        self.human_mask_downsample = 16
        self.heat_map_scale = cfg.heat_map_size
        self.view_per_image = 64

    def statistic_crop(self):
        overlap_thresh = 0.1
        total_cnt = 0
        nonhuman_cnt = 0
        nonhuman_score = []
        for image_name in self.human_bboxes.keys():
            crop = self.annotations[image_name]['bbox']
            score = self.annotations[image_name]['score']
            hbox = self.human_bboxes[image_name]
            crop = np.array(crop).reshape(-1, 4)
            hbox = np.array(hbox).reshape(-1, 4)
            score = np.array(score).reshape(-1)
            overlap = compute_overlap(crop, hbox)
            total_cnt += overlap.shape[0]
            nonhuman_cnt += (overlap < overlap_thresh).sum()
            nonhuman_score.extend(score[overlap < overlap_thresh].tolist())
        print('{} human images, {} crops, {} non-human, {:.2%}'.format(
            len(self.human_bboxes.keys()), total_cnt, nonhuman_cnt, float(nonhuman_cnt) / total_cnt
        ))
        nonhuman_score = np.array(nonhuman_score).reshape(-1)
        print('{} scores, mean={:.2f}, median={:.2f}, max={:.2f}, min={:.2f}'.format(
            nonhuman_score.shape[0], np.mean(nonhuman_score), np.median(nonhuman_score),
            np.max(nonhuman_score), np.min(nonhuman_score)
        ))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        rs_width, rs_height = resized_image.size
        ratio_h = float(rs_height) / im_height
        ratio_w = float(rs_width) / im_width
        if image_name in self.human_bboxes:
            hbox = self.human_bboxes[image_name]
            hbox = rescale_bbox(hbox, ratio_w, ratio_h)
        else:
            hbox = np.array([[-1, -1, -1, -1]]).astype(np.float32)
        heat_map_file = os.path.join(self.heat_map_dir, os.path.splitext(image_name)[0] + '.png')
        heat_map = Image.open(heat_map_file)
        # hm_w, hm_h = int(rs_width * self.heat_map_scale), int(rs_height * self.heat_map_scale)
        hm_w = hm_h = 64
        heat_map = heat_map.resize((hm_w, hm_h))
        crop = self.annotations[image_name]['bbox']
        crop = rescale_bbox(crop, ratio_w, ratio_h)
        score = self.annotations[image_name]['score']
        # score = [float(s - MOS_MEAN) / MOS_STD for s in score]
        score = torch.tensor(score).reshape(-1)
        if self.augmentation:
            if random.uniform(0,1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                heat_map  = ImageOps.mirror(heat_map)
                temp_x1 = crop[:, 0].copy()
                crop[:, 0] = rs_width - crop[:, 2]
                crop[:, 2] = rs_width - temp_x1

                if image_name in self.human_bboxes:
                    # print('human mirror_before', hbox[0])
                    temp_x1 = hbox[:,0].copy()
                    hbox[:,0] = rs_width - hbox[:,2]
                    hbox[:,2] = rs_width - temp_x1
                    # print('human mirror after',  hbox)
            resized_image = self.PhotometricDistort(resized_image)
        im = self.image_transformer(resized_image)
        heat_map = self.heat_map_transformer(heat_map)
        # crop_mask = generate_crop_mask(crop, rs_width, rs_height,
        #                                self.crop_mask_downsample)
        crop_mask = generate_target_size_crop_mask(crop, rs_width, rs_height, 64, 64)
        partition_mask = generate_partition_mask(hbox, rs_width, rs_height,
                                                 self.human_mask_downsample)
        return im, crop, hbox, heat_map, crop_mask, partition_mask, score, im_width, im_height

def compute_overlap(crops, human_bbox):
    human_bbox = human_bbox.reshape(-1)
    if not isinstance(crops, np.ndarray):
        crops = np.array(crops).reshape((-1,4))
    over_x1 = np.maximum(crops[:,0], human_bbox[0])
    over_x2 = np.minimum(crops[:,2], human_bbox[2])
    over_y1 = np.maximum(crops[:,1], human_bbox[1])
    over_y2 = np.minimum(crops[:,3], human_bbox[3])

    over_w  = np.maximum(0, over_x2 - over_x1)
    over_h  = np.maximum(0, over_y2 - over_y1)
    over_area = over_w * over_h
    overlap = over_area / ((human_bbox[2] - human_bbox[0]) * (human_bbox[3] - human_bbox[1]))
    return overlap

def count_GAICD():
    human_bboxes = json.load(open(cfg.GAIC_human, 'r'))
    human_lists = list(human_bboxes.keys())
    human_split = dict()
    data_split = dict()

    GAICD_path = '/workspace/aesthetic_cropping/dataset/GAICD/images'
    assert os.path.exists(GAICD_path), GAICD_path
    for split in ['train', 'test']:
        subpath = os.path.join(GAICD_path, split)
        data_split[split] = os.listdir(subpath)
        humans = [im for im in data_split[split] if im in human_lists]
        human_split[split] = humans
        print('{} set {} images, {} human-centric images'.format(split, len(data_split[split]), len(human_split[split])))
    with open(os.path.join(cfg.GAIC_path, 'original_data_split.json'), 'w') as f:
        json.dump(data_split,f)
    with open(os.path.join(cfg.GAIC_path, 'human_data_split.json'), 'w') as f:
        json.dump(human_split,f)

if __name__ == '__main__':
    cpc_dataset = CPCDataset(only_human_images=False, keep_aspect_ratio=False)
    cpc_dataset.statistic_crop()
    print('CPC Dataset contains {} images'.format(len(cpc_dataset)))
    dataloader = DataLoader(cpc_dataset, batch_size=1, num_workers=0, shuffle=False)
    for batch_idx, data in enumerate(dataloader):
        im, crop, hbox, heat_map, crop_mask, partition_mask, score, im_width, im_height = data
        print(im.shape, crop.shape, hbox.shape, heat_map.shape, crop_mask.shape,
              partition_mask.shape, score.shape, im_width.shape, im_height.shape)

    # fcdb_testset = FCDBDataset(only_human_images=False, keep_aspect_ratio=True)
    # print('FCDB testset has {} images'.format(len(fcdb_testset)))
    # dataloader = DataLoader(fcdb_testset, batch_size=1, num_workers=4)
    # for batch_idx, data in enumerate(dataloader):
    #     im, crop, partition_mask, w, h = data
    #     print(im.shape, crop.shape, partition_mask.shape, w.shape, h.shape)
    #
    # FLMS_testset = FLMSDataset(keep_aspect_ratio=True, only_human_images=False)
    # print('FLMS testset has {} images'.format(len(FLMS_testset)))
    # dataloader = DataLoader(FLMS_testset, batch_size=1, num_workers=4)
    # for batch_idx, data in enumerate(dataloader):
    #     im, crop, partition_mask, w, h = data
    #     print(im.shape, crop.shape, partition_mask.shape, w.shape, h.shape)

    # GAICD_testset = GAICDataset(only_human_images=False, keep_aspect_ratio=True, split='train')
    # print('GAICD testset has {} images'.format(len(GAICD_testset)))
    # dataloader = DataLoader(GAICD_testset, batch_size=1, num_workers=0, shuffle=False)
    # for batch_idx, data in enumerate(dataloader):
    #     im, crop, heat_map, crop_mask, partition_mask, score, im_width, im_height = data
    #     print(im.shape, crop.shape, heat_map.shape, crop_mask.shape,
    #           partition_mask.shape, score.shape, im_width.shape, im_height.shape)
    #     print(crop)
    #     print(score)
    # GAICD_testset.statistic_crop()