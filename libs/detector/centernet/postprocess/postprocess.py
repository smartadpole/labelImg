#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: postprocess.py
@time: 2021/4/6 上午11:54
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../../'))

from yacs.config import CfgNode as CN
import torch
from libs.detector.utils.python_nms import centernet_nms
from libs.detector.centernet.utils import get_affine_transform

IMAGE_SIZE = 320

MODEL = CN()
MODEL.CENTER_VARIANCE = 0.1
MODEL.SIZE_VARIANCE = 0.2

NUM_CLASSES = 1

import numpy as np

NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.4

def _gather_feat(feat, ind):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = torch.from_numpy(feat)
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    feat = feat.data.cpu().numpy()
    return feat

def _topk(scores, K=100):
    batch, cat, height, width = scores.shape
    scores = torch.from_numpy(scores)

    topk_scores, topk_inds = torch.topk(scores.reshape(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys.data.cpu().numpy(), topk_xs.data.cpu().numpy()


def ctdet_decode(heat, wh, reg=None, K=100):
    batch, cat, height, width = heat.shape

    # perform nms on heatmaps
    heat = centernet_nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.reshape(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], 2)
    detections = np.concatenate([bboxes, scores, clses], 2)

    return detections

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret

def post_process(dets, meta, scale=1):
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], NUM_CLASSES)
    for j in range(1, NUM_CLASSES + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def PostProcessor(output, meta, K=100):

    output = output[-1]
    hm = output[:, 0:1, :, :]
    wh = output[:, 3:5, :, :]
    reg = output[:, 1:3, :, :]

    dets = ctdet_decode(hm, wh, reg=reg, K=K)
    dets = post_process(dets, meta)

    return dets
