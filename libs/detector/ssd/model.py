#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: 1054568970@.com
@file: model.py
@time: 2021/4/14 10:42
@desc:
'''
import os
from libs.detector.ssd.onnxmodel import ONNXModel
from libs.detector.ssd.preprocess import pre_process
from libs.detector.ssd.postprocess.ssd import PostProcessor_SSD
from libs.detector.ssd.postprocess.ssd import THRESHOLD

class SSD(object):
    def __init__(self, file='./config/cleaner/ssd.onnx'):
        if os.path.isfile(file):
            self.net = ONNXModel(file)
        else:
            raise IOError("no such file {}".format(self.model_file_4_detect))

    def forward(self, image, class_names_4_detect):
        image, ratio = pre_process(image)
        out = self.net.forward(image)
        results_batch = PostProcessor_SSD(out[0], out[1], out[2])

        # TODO : get rect
        shapes = []
        for result in results_batch:
            if len(result) > 0:
                result = result.tolist()
                for r in result:
                    x, y, x2, y2, label, score = r
                    x, y, x2, y2, label = int(x) * ratio[0], int(y) * ratio[1], int(x2) * ratio[0], int(y2) * ratio[
                        1], int(label)
                    if score < THRESHOLD[label] or label == 0:
                        continue

                    shapes.append(
                        (class_names_4_detect[label - 1], [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False))

        return shapes
