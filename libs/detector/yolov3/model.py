#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 刘家兴
@contact: ljx0ml@163.com
@file: model.py
@time: 2021/8/3 12.02
@desc:
'''
import os
from libs.detector.ssd.onnxmodel import ONNXModel
from libs.detector.yolov3.preprocess import pre_process as yoloPreProcess
from libs.detector.yolov3.postprocess.postprocess import IMAGE_SIZE_YOLOV3, THRESHOLD_YOLOV3, post_processing,\
    load_class_names,plot_boxes_cv2
import cv2
import onnxruntime

class YOLOv3(object):
    def __init__(self, file='./config/i18R/yolov3.onnx',class_sel=[]):
        self.class_sel=class_sel
        self.classes = load_class_names("config/class.names")
        if os.path.isfile(file):
            self.session = onnxruntime.InferenceSession(file)
        else:
            raise IOError("no such file {}".format(file))

    def forward(self, image):
        oriY = image.shape[0]
        oriX = image.shape[1]
        image = cv2.resize(image, (IMAGE_SIZE_YOLOV3, IMAGE_SIZE_YOLOV3), interpolation=cv2.INTER_LINEAR)
        image = yoloPreProcess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: image})
        boxes = post_processing(image, THRESHOLD_YOLOV3, 0.6, outputs)


        # # TODO : get rect
        shapes = []
        for result in boxes:
            if len(result) > 0:
                result = result
                result = [r for r in result if r[4] > THRESHOLD_YOLOV3]
                for r in result:
                    x, y, x2, y2, score, score1, label = r

                    if not self.classes[label] in self.class_cl:
                        continue

                    y = y * oriY
                    y2 = y2 * oriY
                    x = x * oriX
                    x2 = x2 * oriX
                    x, y, x2, y2, score, label = int(x), int(y), int(x2), int(y2), float(score), int(label)
                    shapes.append((self.classes[label], [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False, 0))

        return shapes
