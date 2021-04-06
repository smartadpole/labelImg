#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: preprocess.py
@time: 2021/4/6 下午12:03
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../../'))

import cv2
import numpy as np
from libs.detector.centernet.utils import get_affine_transform

mean = [0.408, 0.447, 0.470]
std = [0.289, 0.274, 0.278]


def pre_process(image, size, scale, meta=None, down_ratio=4):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    inp_height, inp_width = size, size
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    # images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}

    return images, meta