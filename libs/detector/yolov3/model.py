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
from libs.detector.yolov3.preprocess import preProcessPadding
from libs.detector.yolov3.postprocess.postprocess import IMAGE_SIZE_YOLOV3, THRESHOLD_YOLOV3, post_processing,\
    load_class_names,plot_boxes_cv2
import cv2
import onnxruntime
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)).split('libs')[0]

class YOLOv3(object):
    def __init__(self, file='./config/i18R/yolov3.onnx',class_sel=[]):
        self.classes = load_class_names(CURRENT_DIR+"config/i18R/classes.names")
        self.class_sel = class_sel

        if os.path.isfile(file):
            self.session = onnxruntime.InferenceSession(file)
        else:
            raise IOError("no such file {}".format(file))

    def forward(self, image):
        oriY = image.shape[0]
        oriX = image.shape[1]
        image = preProcessPadding(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: image})
        boxes = post_processing(image, THRESHOLD_YOLOV3, 0.45, outputs)


        # # TODO : get rect
        shapes = []
        results_box=[]
        for result in boxes:
            if len(result) > 0:
                result = result
                result = [r for r in result if r[4] > THRESHOLD_YOLOV3]
                for r in result:
                    x, y, x2, y2, score, score1, label = r

                    if int(label) > len(self.classes)-1 or self.classes[int(label)] not in self.class_sel:
                        continue

                    w = IMAGE_SIZE_YOLOV3 / max(oriX,oriY)

                    y = (y * IMAGE_SIZE_YOLOV3 - (IMAGE_SIZE_YOLOV3-w*oriY)/2)/w
                    y2 = (y2 * IMAGE_SIZE_YOLOV3 - (IMAGE_SIZE_YOLOV3-w*oriY)/2)/w
                    x = (x * IMAGE_SIZE_YOLOV3 - (IMAGE_SIZE_YOLOV3-w*oriX)/2)/w
                    x2 = (x2 * IMAGE_SIZE_YOLOV3 - (IMAGE_SIZE_YOLOV3-w*oriX)/2)/w

                    x, y, x2, y2, score, label = int(x), int(y), int(x2), int(y2), float(score), int(label)
                    shapes.append((self.classes[label], [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False, 0))
                    results_box.append([x, y, x2, y2, score, self.classes[label]])
        return shapes,results_box
