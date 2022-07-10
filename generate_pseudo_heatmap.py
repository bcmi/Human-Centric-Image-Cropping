import os
import cv2
import numpy as np
import pickle
import lmdb
import datetime
import torch
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import math
import shutil
from tqdm import tqdm

from config_GAICD import cfg

heat_map_dir = './heat_map_gt'
os.makedirs(heat_map_dir, exist_ok=True)

class HeatMap:
    def __init__(self, dataset, top_k=10):
        self.dataset = dataset
        self.topk = top_k
        self.weighted_sum = True
        if self.dataset == 'CPC':
            self.image_dir = cfg.CPC_image
            self.human_bboxes = json.load(open(cfg.CPC_human, 'r'))
            self.annotations = json.load(open(cfg.CPC_anno, 'r'))
        else:
            self.image_dir = cfg.GAIC_image
            self.human_bboxes = json.load(open(cfg.GAIC_human, 'r'))
            self.annotations = json.load(open(cfg.GAIC_anno, 'r'))
        assert os.path.exists(self.image_dir), self.image_dir
        self.image_list = list(self.annotations.keys())
        self.human_image_list = list(self.human_bboxes.keys())
        self.heat_map_path = os.path.join(heat_map_dir, self.dataset)
        if os.path.exists(self.heat_map_path):
            shutil.rmtree(self.heat_map_path)
        os.makedirs(self.heat_map_path, exist_ok=True)
        self.display_path = os.path.join(self.heat_map_path, 'display')
        self.mask_path    = os.path.join(self.heat_map_path, 'mask')
        os.makedirs(self.display_path)
        os.makedirs(self.mask_path)
        self.score_mean, self.score_std = self.statistic_score()
        self.generate_heat_map()

    def statistic_score(self):
        score_list = []
        for image_name in self.image_list:
            if self.dataset == 'CPC':
                crop = self.annotations[image_name]['bboxes']
                score = self.annotations[image_name]['scores']
            else:
                crop = self.annotations[image_name]['bbox']
                score = self.annotations[image_name]['score']
            score_list.extend(score)
        score = np.array(score_list).reshape(-1)
        mean  = np.mean(score)
        std   = np.std(score)
        print('{} has {} score annotations, mean={:.2f}, std={:.2f}'.format(
            self.dataset, len(score), mean, std))
        return mean, std

    def normalize_mask(self, mask, score, weighted_sum=False):
        height, width = mask.shape[1:]
        if not weighted_sum:
            mask = np.sum(mask, axis=0)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
        # else:
        #     weights = (score - score.min()) / (score.max() - score.min())
        #     weighted_mask = np.sum(weights[:, None, None] * mask, axis=0)
        #     weighted_mask = (weighted_mask - weighted_mask.min()) / (weighted_mask.max() - weighted_mask.min())
        #     return weighted_mask
        else:
            if len(mask) > 10:
                pos = score > self.score_mean
                mask = mask[pos]
                score = score[pos]
            area  = mask.sum(2).sum(1) / (height * width)
            score = score + area * 2
            weight = torch.softmax(torch.from_numpy(score).reshape(-1, 1), dim=0).unsqueeze(2).numpy()
            pos_mask = np.sum(weight * mask, axis=0)
            mask = (pos_mask - pos_mask.min()) / (pos_mask.max() - pos_mask.min())
        exp_mask = np.expand_dims(mask, axis=2)
        norm_mask = np.zeros((height, width, 1))
        cv2.normalize(exp_mask, norm_mask, 0, 255, cv2.NORM_MINMAX)
        norm_mask = np.asarray(norm_mask, dtype=np.uint8)
        return norm_mask

    def mask2hotmap(self, src, mask):
        heat_im = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heat_im = cv2.cvtColor(heat_im, cv2.COLOR_BGR2RGB)
        fuse_im = cv2.addWeighted(src, 0.3, heat_im, 0.7, 0)
        return fuse_im

    def generate_heat_map(self):
        for image_name in tqdm(self.image_list):
            image_file = os.path.join(self.image_dir, image_name)
            assert os.path.exists(image_file), image_file
            src = cv2.imread(image_file)
            height, width = src.shape[0:2]

            if self.dataset == 'CPC':
                crop = self.annotations[image_name]['bboxes']
                score = self.annotations[image_name]['scores']
            else:
                crop = self.annotations[image_name]['bbox']
                score = self.annotations[image_name]['score']

            crop = np.array(crop).astype(np.int).reshape((-1, 4))
            score = np.array(score).reshape(-1)
            rank = np.argsort(score)[::-1]
            crop = crop[rank]
            score = score[rank]
            topk = min(self.topk, len(crop))
            mask = np.zeros((len(crop), height, width))
            for i in range(len(crop)):
                x1,y1,x2,y2 = [int(c) for c in crop[i]]
                mask[i, y1:y2, x1:x2] += 1
            fuse_mask = self.normalize_mask(mask, score, weighted_sum=self.weighted_sum)
            mask_path = os.path.join(self.mask_path, image_name.replace('.jpg', '.png'))
            cv2.imwrite(mask_path, fuse_mask)

            if image_name not in self.human_image_list:
                continue
            plt.figure(figsize=(10,10))
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.subplot(2,3,1)
            plt.imshow(src[:,:,::-1])
            plt.title('input image')
            plt.axis('off')

            fuse_im = self.mask2hotmap(src, self.normalize_mask(mask[:5], score[:5], weighted_sum=self.weighted_sum))
            plt.subplot(2,3,2)
            plt.imshow(fuse_im, vmin=0, vmax=255)
            plt.title('top-5 heat map')
            plt.axis('off')

            fuse_im = self.mask2hotmap(src, self.normalize_mask(mask[:10], score[:10], weighted_sum=self.weighted_sum))
            plt.subplot(2, 3, 3)
            plt.imshow(fuse_im, vmin=0, vmax=255)
            plt.title('top-10 heat map')
            plt.axis('off')

            top_mask = mask
            fuse_im = self.mask2hotmap(src, self.normalize_mask(top_mask, score, weighted_sum=self.weighted_sum))
            plt.subplot(2, 3, 4)
            plt.imshow(fuse_im, vmin=0, vmax=255)
            plt.title('score > mean_score heat map'.format(mask.shape[0]))
            plt.axis('off')

            best_crop = [int(x) for x in crop[0]]
            best_im = fuse_im[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
            plt.subplot(2,3,5)
            plt.imshow(best_im)
            plt.title('best crop')
            plt.axis('off')

            bad_crop = [int(x) for x in crop[-1]]
            bad_im = fuse_im[bad_crop[1] : bad_crop[3], bad_crop[0] : bad_crop[2]]
            plt.subplot(2, 3, 6)
            plt.imshow(bad_im)
            plt.title('worst crop')
            plt.axis('off')

            plt.tight_layout()
            save_fig = os.path.join(self.display_path, image_name)
            plt.savefig(save_fig)
            # plt.show()
            plt.close()
            # print(save_fig, human_numer, nonhuman_number)


if __name__ == '__main__':
    HeatMap('GAICD')