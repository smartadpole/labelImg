#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: 1054568970@.com
@file: model.py
@time: 2021/4/14 10:20
@desc:
'''
import numpy as np
import cv2
from libs.detector.ssd.postprocess.ssd import PIXEL_MEAN

def pre_process(image):
    h, w, _ = image.shape
    image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_CUBIC)
    image = (image - PIXEL_MEAN).astype(np.float32).transpose((2, 0, 1))[np.newaxis, ::]

    return image, (w / 320, h / 320)