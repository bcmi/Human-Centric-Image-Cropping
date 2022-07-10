import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
import sys
sys.path.insert(0, './..')
'''
install RoIAlign and RoIAlign libraries following the repository:
https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch 
'''
from roi_align.modules.roi_align import RoIAlignAvg, RoIAlign
from rod_align.modules.rod_align import RoDAlignAvg, RoDAlign

import os
import torchvision.transforms as transforms
from PIL import Image
import math
import numpy as np

class vgg_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path=None):
        super(vgg_base, self).__init__()

        vgg = models.vgg16(pretrained=loadweights)

        # if downsample == 4:
        #     self.feature = nn.Sequential(vgg.features[:-1])
        # elif downsample == 5:
        #     self.feature = nn.Sequential(vgg.features)

        self.feature3 = nn.Sequential(vgg.features[:23])
        self.feature4 = nn.Sequential(vgg.features[23:30])
        self.feature5 = nn.Sequential(vgg.features[30:])

        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5

class resnet50_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path=None):
        super(resnet50_base, self).__init__()

        resnet50 = models.resnet50(pretrained=loadweights)
        self.feature3 = nn.Sequential(resnet50.conv1,resnet50.bn1,resnet50.relu,resnet50.maxpool,resnet50.layer1,resnet50.layer2)
        self.feature4 = nn.Sequential(resnet50.layer3)
        self.feature5 = nn.Sequential(resnet50.layer4)

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5

class PartitionAwareModule(nn.Module):
    def __init__(self, in_dim, partiition=9, concat_with_human=True):
        super(PartitionAwareModule, self).__init__()
        alignsize  = 3
        downsample = 4
        human_dim  = in_dim // 2 if concat_with_human else 0
        if partiition in [0,9]:
            self.group_conv = nn.Conv2d(in_dim + human_dim, in_dim * 9, kernel_size=3, stride=1, padding=1)
        else:
            self.group_conv = nn.Conv2d(in_dim + human_dim, in_dim * partiition, kernel_size=3, stride=1, padding=1)
        if concat_with_human:
            self.RoIAlign  = RoIAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
            self.RoIConv   = nn.Conv2d(in_dim, human_dim, kernel_size=3)
        self.partition = partiition
        self.concat_with_human = concat_with_human

    def forward(self, x, human_box, partition_mask):
        # x: (b,c,h,w)
        # p_mask: # b,9,1,h,w
        if self.concat_with_human:
            humanRoI = self.RoIAlign(x, human_box)
            humanRoI = self.RoIConv(humanRoI)
            humanRoI = repeat(humanRoI, 'b c 1 1 -> b c h w', h=x.shape[-2], w=x.shape[-1])
            cat_feat = torch.cat([x, humanRoI], dim=1)
            p_conv = F.relu(self.group_conv(cat_feat))
        else:
            p_conv = F.relu(self.group_conv(x))
        if self.partition in [0,9]:
            p_feat = torch.chunk(p_conv, 9, dim=1)  # tuple, each of shape (b,c,h,w)
            p_feat = torch.stack(p_feat, dim=1)  # b,9,c,h,w
            p_mean = torch.mean(p_feat, dim=2)
            if self.partition == 0:
                fused = torch.mean(p_feat, dim=1)
            else:
                fused  = torch.sum(p_feat * partition_mask.unsqueeze(2), dim=1)
        else:
            human_mask = partition_mask[:,4].unsqueeze(1) # b,1,h,w
            if self.partition == 1:
                fused  = p_conv * human_mask
                p_mean = torch.mean(p_conv, dim=1).unsqueeze(1) # b,1,h,w
            else:
                p_feat = torch.chunk(p_conv, self.partition, dim=1)  # tuple, each of shape (b,c,h,w)
                p_feat = torch.stack(p_feat, dim=1)  # b,2,c,h,w
                p_mean = torch.mean(p_feat,  dim=2)  # b,2,h,w
                non_mask = 1 - human_mask
                cat_mask = torch.stack([human_mask, non_mask], dim=1) # b,2,1,h,w
                fused  = torch.sum(cat_mask * p_feat, dim=1) # b,c,h,w
        out = x + fused
        return out, p_mean

class ContentAwareModule(nn.Module):
    def __init__(self, in_dim):
        super(ContentAwareModule, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                nn.ReLU(True))
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                nn.ReLU(True))
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_dim, 1, kernel_size=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        out = self.conv3(x)
        return out

class ContentPreserveModule(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim, mode='gcn'):
        super(ContentPreserveModule, self).__init__()
        if mode == 'conv':
            self.relation_encoder = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_dim, inter_dim, kernel_size=1))
        else:
            self.relation_encoder = GraphResoningModule(in_dim, in_dim, inter_dim)

        self.preserve_predict = nn.Sequential(
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inter_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.content_feat_layer = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(1),
            nn.Linear(5408, out_dim)
        )

    def forward(self, feat_map, crop_mask):
        '''
        :param feat_map: b,c,h,w
        :param crop_mask: b,n,h,w
        :return:
        '''
        relation_feat = self.relation_encoder(feat_map)
        heat_map = self.preserve_predict(relation_feat)
        B, N, H, W = crop_mask.shape
        rep_heat = repeat(heat_map, 'b 1 h w -> b n h w', n=N)
        cat_map = torch.stack([rep_heat, crop_mask], dim=2)
        cat_map = rearrange(cat_map, 'b n c h w -> (b n) c h w', b=B, n=N)
        content_feat = self.content_feat_layer(cat_map)
        return heat_map, content_feat

class GraphResoningModule(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim):
        super(GraphResoningModule, self).__init__()
        self.phi   = nn.Linear(in_dim, inter_dim, bias=False)
        self.theta = nn.Linear(in_dim, inter_dim, bias=False)
        self.weight= nn.Parameter(torch.empty(in_dim, out_dim))
        init.xavier_uniform_(self.weight.data)

    def forward(self, feat_map):
        '''
        :param feat_map: b,c,h,w
        :param crop_mask: b,n,h,w
        :return:
        '''
        B,D,H,W = feat_map.shape
        feat_vec = rearrange(feat_map, 'b c h w -> b (h w) c')
        phi      = self.phi(feat_vec) # b,n,d
        theta    = self.theta(feat_vec)  # b,n,d
        product  = torch.matmul(phi, theta.permute(0,2,1)) # b,n,n
        phi_norm = torch.norm(phi, p=2, dim=-1, keepdim=True) # b,n,1
        theta_norm = torch.norm(theta, p=2, dim=-1, keepdim=True) # b,n,1
        norms    = torch.matmul(phi_norm, theta_norm.permute(0,2,1)) # b,n,n
        sim      = torch.softmax(product / norms, dim=-1) # b,n,n
        out      = torch.matmul(sim, feat_vec) # b,n,d
        out      = torch.matmul(out, self.weight) # b,n,d'
        updated  = rearrange(out, 'b (h w) d -> b d h w', h=H, w=W)
        updated  = F.relu(updated)
        return updated

class HumanCentricCroppingModel(nn.Module):
    def __init__(self, loadweights=True, cfg=None):
        super(HumanCentricCroppingModel, self).__init__()
        if cfg is None:
            from config_GAICD import cfg
        reduce_dim = cfg.reduced_dim
        alignsize  = 9
        cp_dim     = 256
        downsample = 4
        self.cfg = cfg
        self.backbone = vgg_base(loadweights=loadweights)
        self.f3_layer = nn.Conv2d(512, 256, kernel_size=1)
        self.f4_layer = nn.Conv2d(512, 256, kernel_size=1)
        self.f5_layer = nn.Conv2d(512, 256, kernel_size=1)
        self.feat_ext = nn.Conv2d(256, reduce_dim, kernel_size=1, padding=0)
        if cfg.use_partition_aware:
            self.partition_aware = PartitionAwareModule(reduce_dim, partiition=cfg.partition_aware_type,
                                                        concat_with_human=cfg.concat_with_human)
        else:
            self.partition_aware = None

        if cfg.use_content_preserve:
            self.content_preserve = ContentPreserveModule(reduce_dim, 8, cp_dim, mode=cfg.content_preserve_type)
        else:
            self.content_preserve = None

        fc1_dim = 512
        fc2_dim = 256
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        reduce_dim *= 2
        self.roi_feat_layer = nn.Sequential(
            nn.Conv2d(reduce_dim, fc1_dim, kernel_size=alignsize),
            nn.ReLU(True),
            nn.Flatten(1),
            nn.Linear(fc1_dim, fc2_dim, bias=False),
        )
        if cfg.use_content_preserve and not cfg.only_content_preserve:
            self.fc_layer = nn.Linear(fc2_dim+cp_dim, 1)
        else:
            self.fc_layer = nn.Linear(fc2_dim, 1)

    def forward(self, im, crop_box, human_box, crop_mask, human_mask):
        B, N, _ = crop_box.shape
        if crop_box.shape[-1] == 4:
            index = torch.arange(B).view(-1, 1).repeat(1, N).reshape(B, N, 1).to(im.device)
            crop_box  = torch.cat((index, crop_box), dim=-1).contiguous()
        if human_box.shape[-1] == 4:
            hindex    = torch.arange(B).view(-1, 1, 1).to(im.device)
            human_box = torch.cat((hindex, human_box), dim=-1).contiguous()
        if crop_box.dim() == 3:
            crop_box  = crop_box.reshape(-1, 5)
            human_box = human_box.reshape(-1, 5)

        f3, f4, f5 = self.backbone(im)
        f3 = F.interpolate(f3, size=f4.shape[2:], mode='bilinear', align_corners=True)
        f5 = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=True)
        agg_feat = self.f3_layer(f3) + self.f4_layer(f4) + self.f5_layer(f5)
        red_feat = self.feat_ext(agg_feat)
        contain_human = (torch.count_nonzero(human_mask[0,4]) > 0)
        part_feat = torch.zeros_like(human_mask.detach())

        if self.partition_aware and contain_human:
            red_feat, part_feat = self.partition_aware(red_feat, human_box, human_mask)

        RoI_feat = self.RoIAlign(red_feat, crop_box)
        RoD_feat = self.RoDAlign(red_feat, crop_box)
        cat_feat = torch.cat([RoI_feat, RoD_feat], dim=1)
        crop_feat = self.roi_feat_layer(cat_feat)
        heat_map = torch.zeros_like(crop_mask[:, 0:1])

        if self.content_preserve:
            heat_map, cont_feat = self.content_preserve(red_feat, crop_mask)
            if self.cfg.only_content_preserve:
                crop_feat = cont_feat
            else:
                crop_feat = torch.cat([crop_feat, cont_feat], dim=1)

        crop_score = self.fc_layer(crop_feat)
        score = rearrange(crop_score, '(b n) 1 -> b n', b=B, n=N)
        return part_feat, heat_map, score

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

def listwise_view_ranking_loss(pre_score, gt_score):
    if pre_score.dim() > 1:
        pre_score = pre_score.reshape(-1)
    if gt_score.dim() > 1:
        gt_score  = gt_score.reshape(-1)
    assert pre_score.shape == gt_score.shape, '{} vs. {}'.format(pre_score.shape, gt_score.shape)
    scores = nn.LogSoftmax(dim=-1)(pre_score)
    score_list  = gt_score.detach().cpu().numpy().tolist()
    sort_scores = torch.sort(torch.unique(gt_score))[0].detach().cpu().numpy().tolist()
    label_list  = [sort_scores.index(e) + 1 for e in score_list]
    labels = torch.tensor(label_list).float().to(gt_score.device)
    labels = F.softmax(labels, dim=-1)
    loss = torch.sum(-(scores * labels), dim=-1)
    return loss

def score_weighted_regression_loss(pre_score, gt_score, score_mean):
    if pre_score.dim() > 1:
        pre_score = pre_score.reshape(-1)
    if gt_score.dim() > 1:
        gt_score  = gt_score.reshape(-1)
    assert pre_score.shape == gt_score.shape, '{} vs. {}'.format(pre_score.shape, gt_score.shape)
    l1_loss = F.smooth_l1_loss(pre_score, gt_score, reduction='none')
    weight  = torch.exp((gt_score - score_mean).clip(min=0,max=100))
    reg_loss= torch.mean(weight * l1_loss)
    return reg_loss

def score_regression_loss(pre_score, gt_score):
    if pre_score.dim() > 1:
        pre_score = pre_score.reshape(-1)
    if gt_score.dim() > 1:
        gt_score  = gt_score.reshape(-1)
    assert pre_score.shape == gt_score.shape, '{} vs. {}'.format(pre_score.shape, gt_score.shape)
    l1_loss = F.smooth_l1_loss(pre_score, gt_score, reduction='mean')
    return l1_loss

def score_rank_loss(pre_score, gt_score):
    if pre_score.dim() > 1:
        pre_score = pre_score.reshape(-1)
    if gt_score.dim() > 1:
        gt_score  = gt_score.reshape(-1)
    assert pre_score.shape == gt_score.shape, '{} vs. {}'.format(pre_score.shape, gt_score.shape)
    N = pre_score.shape[0]
    pair_num = N * (N-1) / 2
    pre_diff = pre_score[:,None] - pre_score[None,:]
    gt_diff  = gt_score[:,None]  - gt_score[None,:]
    indicat  = -1 * torch.sin(gt_diff) * (pre_diff - gt_diff)
    diff     = torch.maximum(indicat, torch.zeros_like(indicat))
    rank_loss= torch.sum(diff) / pair_num
    return rank_loss

def partition_ce_loss(pre_patition, gt_partition):
    '''
    :param pre_patition: b,9,h,w
    :param gt_partition: b,9,h,w
    :return:
    '''
    pre_logit = rearrange(pre_patition, 'b c h w -> (b h w) c')
    gt_mask   = rearrange(gt_partition, 'b c h w -> (b h w) c')
    gt_label  = torch.argmax(gt_mask, dim=-1) # (b h w)
    loss      = F.cross_entropy(pre_logit, gt_label)
    return loss

if __name__ == '__main__':
    device = torch.device('cuda:0')
    net = HumanCentricCroppingModel(loadweights=True).to(device)
    w,h = 256, 256
    x = torch.randn(2,3,h,w).to(device)
    boxes = torch.tensor([[64, 64, 223, 223],
                          [64, 64, 223, 223]]).float().to(device)
    boxes = boxes.unsqueeze(0).repeat(2,1,1)
    human_box = torch.tensor([32, 32, 64, 64]).reshape(1,1,-1).float().to(device)
    human_box = human_box.repeat(2,1,1)
    crop_mask = torch.randn(2, 2,h//4,w//4).to(device)
    human_mask = torch.randn(2,9,h//16,w//16).to(device)
    part_mask, heat_map, score = net(x, boxes, human_box, crop_mask, human_mask)
    print(part_mask.shape, heat_map.shape, score.shape)
    print(score)