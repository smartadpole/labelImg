#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: 1054568970@.com
@file: model.py
@time: 2021/4/14 10:56
@desc:
'''
import os
from libs.detector.ssd.onnxmodel import ONNXModel
from libs.detector.yolov5.preprocess import pre_process as yoloPreProcess
from libs.detector.yolov5.postprocess.postprocess import PostProcessor_YOLOV5
from libs.detector.yolov5.postprocess.postprocess import IMAGE_SIZE_YOLOV5, THRESHOLD_YOLOV5
import cv2

class YOLOv5(object):
    def __init__(self, file='./config/human/yolov5.onnx'):
        if os.path.isfile(file):
            self.net = ONNXModel(file)
        else:
            raise IOError("no such file {}".format(self.model_file_4_detect))

    def forward(self, image):
        oriY = image.shape[0]
        oriX = image.shape[1]
        image = cv2.resize(image, (IMAGE_SIZE_YOLOV5, IMAGE_SIZE_YOLOV5))
        image = yoloPreProcess(image)
        out = self.net.forward(image)
        results_batch = PostProcessor_YOLOV5(out)

        # TODO : get rect
        shapes = []
        for result in results_batch:
            if len(result) > 0:
                result = result.tolist()
                result = [r for r in result if r[4] > THRESHOLD_YOLOV5]
                for r in result:
                    x, y, x2, y2, score, label = r
                    if label != 0:  # detect humans only
                        continue

                    y = y / IMAGE_SIZE_YOLOV5 * oriY
                    y2 = y2 / IMAGE_SIZE_YOLOV5 * oriY
                    x = x / IMAGE_SIZE_YOLOV5 * oriX
                    x2 = x2 / IMAGE_SIZE_YOLOV5 * oriX
                    x, y, x2, y2, score, label = int(x), int(y), int(x2), int(y2), float(score), int(label)
                    shapes.append(("person", [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False))

        return shapes
