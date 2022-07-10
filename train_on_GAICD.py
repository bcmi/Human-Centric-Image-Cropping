import os
import sys
import numpy as np
from tensorboardX import SummaryWriter
import torch
import time
import datetime
import csv
from tqdm import tqdm
import shutil
import pickle
from scipy.stats import spearmanr
import random
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import math

from cropping_model import HumanCentricCroppingModel
from cropping_model import score_regression_loss, score_rank_loss, \
    score_weighted_regression_loss, listwise_view_ranking_loss
from cropping_dataset import GAICDataset
from config_GAICD import cfg
from test import evaluate_on_GAICD

device = torch.device('cuda:{}'.format(cfg.gpu_id))
torch.cuda.set_device(cfg.gpu_id)
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
MOS_MEAN = 2.95
MOS_STD  = 0.8

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def create_dataloader():
    assert cfg.training_set == 'GAICD', cfg.training_set
    dataset = GAICDataset(only_human_images=cfg.only_human,
                          keep_aspect_ratio=cfg.keep_aspect_ratio,
                          split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=cfg.num_workers,
                                             drop_last=False, worker_init_fn=random.seed(SEED),
                                             pin_memory=True)
    print('training set has {} samples, {} batches'.format(len(dataset), len(dataloader)))
    return dataloader

class Trainer:
    def __init__(self, model):
        self.model = model
        self.epoch = 0
        self.iters = 0
        self.max_epoch = cfg.max_epoch
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.optimizer, self.lr_scheduler = self.get_optimizer()
        self.train_loader = create_dataloader()
        self.eval_results = []
        self.best_results = {'human_srcc': 0, 'human_acc5': 0., 'human_acc10': 0.,
                             'srcc': 0, 'acc5': 0., 'acc10': 0.}
        self.score_loss_type = cfg.loss_type if isinstance(cfg.loss_type, list) else [cfg.loss_type]
        self.l1_loss = torch.nn.L1Loss()

    def get_optimizer(self):
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        if cfg.lr_scheduler == 'cosine':
            warm_up_epochs = 5
            warm_up_with_cosine_lr = lambda epoch: min(1.,(epoch+1) / warm_up_epochs) if epoch <= warm_up_epochs else 0.5 * (
                    math.cos((epoch - warm_up_epochs) / (self.max_epoch - warm_up_epochs) * math.pi) + 1)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warm_up_with_cosine_lr)
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=cfg.lr_decay_epoch, gamma=cfg.lr_decay
            )
        return optim, lr_scheduler

    def run(self):
        print(("========  Begin Training  ========="))
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train()
            if epoch % cfg.eval_freq == 0 or (epoch == self.max_epoch-1):
                self.eval()
                self.record_eval_results()
            self.lr_scheduler.step()

    def visualize_partition_mask(self, im, pre_part, gt_part):
        im       = im.detach().cpu()
        pre_part = torch.softmax(pre_part,dim=0).detach().cpu()
        # pre_part = pre_part.detach().cpu()
        gt_part  = gt_part.detach().cpu()
        im = im * torch.tensor(IMAGE_NET_STD).view(3,1,1) + torch.tensor(IMAGE_NET_MEAN).view(3,1,1)
        trans_fn = transforms.ToPILImage()
        im = trans_fn(im).convert('RGB')
        width, height = im.size

        joint_pre = None
        joint_gt  = None
        for i in range(3):
            h,w = pre_part.shape[1:]
            col_band= torch.ones(h,1).float()
            row_pre = torch.cat([pre_part[i*3], col_band, pre_part[i*3+1], col_band, pre_part[i*3+2]], dim=-1)
            row_gt  = torch.cat([gt_part[i*3],  col_band, gt_part[i*3+1],  col_band, gt_part[i*3+2]],  dim=-1)
            if joint_gt is None:
                joint_gt  = row_gt
                joint_pre = row_pre
            else:
                row_band = torch.ones(1, row_pre.shape[-1]).float()
                joint_pre = torch.cat([joint_pre, row_band, row_pre], dim=0)
                joint_gt  = torch.cat([joint_gt,  row_band, row_gt],  dim=0)
        pre_part = trans_fn(joint_pre).convert('RGB').resize((width,height))
        gt_part  = trans_fn(joint_gt).convert('RGB').resize((width, height))
        ver_band = (np.ones((height,5,3)) * 255).astype(np.uint8)
        cat_im   = np.concatenate([np.asarray(im), ver_band, np.asarray(gt_part), ver_band, np.asarray(pre_part)], axis=1)
        cat_im   = Image.fromarray(cat_im)
        fig_dir  = os.path.join(cfg.exp_path, 'visualization')
        os.makedirs(fig_dir,exist_ok=True)
        fig_file = os.path.join(fig_dir, str(self.iters) + '_part.jpg')
        cat_im.save(fig_file)

    def visualize_heat_map(self, im, pre_heat, gt_heat):
        im       = im.detach().cpu()
        pre_heat = pre_heat.detach().cpu()
        gt_heat  = gt_heat.detach().cpu()
        im = im * torch.tensor(IMAGE_NET_STD).view(3,1,1) + torch.tensor(IMAGE_NET_MEAN).view(3,1,1)
        trans_fn = transforms.ToPILImage()
        im = trans_fn(im).convert('RGB')
        width, height = im.size
        pre_heat = trans_fn(pre_heat).convert('RGB').resize((width,height))
        gt_heat  = trans_fn(gt_heat).convert('RGB').resize((width, height))
        ver_band = (np.ones((height,5,3)) * 255).astype(np.uint8)
        cat_im   = np.concatenate([np.asarray(im), ver_band, np.asarray(gt_heat), ver_band, np.asarray(pre_heat)], axis=1)
        cat_im   = Image.fromarray(cat_im)
        fig_dir  = os.path.join(cfg.exp_path, 'visualization')
        os.makedirs(fig_dir,exist_ok=True)
        fig_file = os.path.join(fig_dir, str(self.iters) + '_content.jpg')
        cat_im.save(fig_file)

    def train(self):
        self.model.train()
        start = time.time()
        batch_loss = 0
        batch_score_loss = 0.
        batch_content_loss = 0.
        total_batch = len(self.train_loader)
        human_cnt = 0.

        for batch_idx, batch_data in enumerate(self.train_loader):
            self.iters += 1
            im        = batch_data[0].to(device)
            rois      = batch_data[1].to(device)
            human_box = batch_data[2].to(device)
            heat_map  = batch_data[3].to(device)
            crop_mask = batch_data[4].to(device)
            part_mask = batch_data[5].to(device)
            score     = batch_data[6].to(device)
            # width     = batch_data[7].to(device)
            # height    = batch_data[8].to(device)
            contain_human = (torch.count_nonzero(part_mask[0, 4]) > 0)

            random_ID = list(range(0, rois.shape[1]))
            random.shuffle(random_ID)
            chosen_ID = random_ID[:64]
            rois = rois[:,chosen_ID]
            crop_mask = crop_mask[:,chosen_ID]
            score = score[:,chosen_ID]

            pre_patition, pred_heat_map, pred_score = self.model(im, rois, human_box, crop_mask, part_mask)
            score_loss = None
            for loss_type in self.score_loss_type:
                if loss_type == 'L1Loss':
                    cur_loss = score_regression_loss(pred_score, score)
                elif loss_type == 'WeightedL1Loss':
                    cur_loss = score_weighted_regression_loss(pred_score, score, MOS_MEAN)
                elif loss_type == 'RankLoss':
                    cur_loss = score_rank_loss(pred_score, score)
                elif loss_type == 'LVRLoss':
                    cur_loss = listwise_view_ranking_loss(pred_score, score)
                else:
                    raise Exception('Undefined score loss type', loss_type)
                if score_loss:
                    score_loss += cur_loss
                else:
                    score_loss = cur_loss
            batch_score_loss += score_loss.item()
            loss = score_loss

            if cfg.use_content_preserve:
                content_loss = self.l1_loss(pred_heat_map.reshape(-1), heat_map.reshape(-1))
                loss += (content_loss * cfg.content_loss_weight)
                batch_content_loss += content_loss.item()
            if contain_human:
                human_cnt += 1
                if human_cnt % cfg.visualize_freq == 0:
                    if cfg.use_partition_aware and cfg.visualize_partition_feature:
                        self.visualize_partition_mask(im[0], pre_patition[0], part_mask[0])
                    if cfg.use_content_preserve and cfg.visualize_heat_map:
                        self.visualize_heat_map(im[0], pred_heat_map[0], heat_map[0])
            batch_loss += loss.item()
            loss = loss / cfg.batch_size
            loss.backward()

            if (batch_idx+1) % cfg.batch_size == 0 or batch_idx >= total_batch-1:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if batch_idx > 0 and batch_idx % cfg.display_freq == 0:
                avg_loss = batch_loss / (1 + batch_idx)
                cur_lr = self.optimizer.param_groups[0]['lr']
                avg_score_loss   = batch_score_loss / (1 + batch_idx)
                self.writer.add_scalar('train/score_loss', avg_score_loss, self.iters)
                self.writer.add_scalar('train/total_loss', avg_loss, self.iters)
                self.writer.add_scalar('train/lr', cur_lr, self.iters)

                if cfg.use_content_preserve:
                    avg_content_loss = batch_content_loss / (1 + batch_idx)
                    self.writer.add_scalar('train/content_loss', avg_content_loss, self.iters)
                else:
                    avg_content_loss = 0.

                time_per_batch = (time.time() - start) / (batch_idx + 1.)
                last_batches = (self.max_epoch - self.epoch - 1) * total_batch + (total_batch - batch_idx - 1)
                last_time = int(last_batches * time_per_batch)
                time_str = str(datetime.timedelta(seconds=last_time))

                print('=== epoch:{}/{}, step:{}/{} | Loss:{:.4f} | Score_Loss:{:.4f} | Content_Loss:{:.4f} | lr:{:.6f} | estimated last time:{} ==='.format(
                    self.epoch, self.max_epoch, batch_idx, total_batch, avg_loss, avg_score_loss, avg_content_loss, cur_lr, time_str
                ))

    def eval(self):
        self.model.eval()
        human_srcc, human_acc5, human_acc10 = evaluate_on_GAICD(self.model, only_human=True)
        srcc, acc5, acc10 = evaluate_on_GAICD(self.model, only_human=False)
        self.eval_results.append([self.epoch, human_srcc, human_acc5, human_acc10,
                                  srcc, acc5, acc10])
        epoch_result = {'human_srcc': human_srcc, 'human_acc5': human_acc5, 'human_acc10': human_acc10,
                        'srcc': srcc, 'acc5': acc5, 'acc10': acc10}
        for m in self.best_results.keys():
            update = False
            if (m != 'disp') and (epoch_result[m] > self.best_results[m]):
                update = True
            elif (m == 'disp') and (epoch_result[m] < self.best_results[m]):
                update = True
            if update:
                self.best_results[m] = epoch_result[m]
                checkpoint_path = os.path.join(cfg.checkpoint_dir, 'best-{}.pth'.format(m))
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Update best {} model, best {}={:.4f}'.format(m, m, self.best_results[m]))
            if m in ['human_srcc', 'srcc']:
                self.writer.add_scalar('test/{}'.format(m), epoch_result[m], self.epoch)
                self.writer.add_scalar('test/best-{}'.format(m), self.best_results[m], self.epoch)
        if self.epoch % cfg.save_freq == 0:
            checkpoint_path = os.path.join(cfg.checkpoint_dir, 'epoch-{}.pth'.format(self.epoch))
            torch.save(self.model.state_dict(), checkpoint_path)

    def record_eval_results(self):
        csv_path = os.path.join(cfg.exp_path, '..', '{}.csv'.format(cfg.exp_name))
        header = ['epoch', 'human_srcc', 'human_acc5', 'human_acc10',
                  'srcc', 'acc5', 'acc10']
        rows = [header]
        for i in range(len(self.eval_results)):
            new_results = []
            for j in range(len(self.eval_results[i])):
                if 'srcc' in header[j]:
                    new_results.append(round(self.eval_results[i][j], 3))
                elif 'acc' in header[j]:
                    new_results.append(round(self.eval_results[i][j], 3))
                else:
                    new_results.append(self.eval_results[i][j])
            self.eval_results[i] = new_results
        rows += self.eval_results
        metrics = [[] for i in header]
        for result in self.eval_results:
            for i, r in enumerate(result):
                metrics[i].append(r)
        for name, m in zip(header, metrics):
            if name == 'epoch':
                continue
            index = m.index(max(m))
            if name == 'disp':
                index = m.index(min(m))
            title = 'best {}(epoch-{})'.format(name, index)
            row = [l[index] for l in metrics]
            row[0] = title
            rows.append(row)
        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)
        print('Save result to ', csv_path)

if __name__ == '__main__':
    cfg.create_path()
    for file in os.listdir('./'):
        if file.endswith('.py'):
            shutil.copy(file, cfg.exp_path)
            print('backup', file)
    net = HumanCentricCroppingModel(loadweights=True, cfg=cfg)
    net = net.to(device)
    trainer = Trainer(net)
    trainer.run()