#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 刘家兴
@contact: ljx0ml@163.com
@file: test.py
@time: 2021/8/3 上午12.02
@desc:
'''
import sys, os

import onnxruntime

from libs.detector.utils.file import Walk, Timer

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../../'))

import argparse
from libs.detector.ssd.onnxmodel import ONNXModel
import time
import cv2
import  numpy as np
from libs.detector.yolov3.postprocess.postprocess import IMAGE_SIZE_YOLOV3, THRESHOLD_YOLOV3, \
    post_processing,plot_boxes_cv2

CLASS_NAMES = [
    'person',
    'escalator', 'escalator_handrails', 'person_dummy', 'escalator_model', 'escalator_handrails_model',
]

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--onnx", type=str, help="onnx model file")
    parser.add_argument("--image", type=str, help="image directory")

    args = parser.parse_args()
    return args

def main():
    args = GetArgs()
    if os.path.isfile(args.image):
        image_paths = [args.image, ]
    else:
        image_paths = Walk(args.image, ["jpg", "png", "jpeg"])

    time_start1 = time.time()
    session = onnxruntime.InferenceSession('./config/i18R/yolov3.onnx')
    time_end2 = time.time()
    print('load model cost', time_end2 - time_start1)

    for i, file in enumerate(sorted(image_paths)):
        timer = Timer()

        timer.Timing("read image")
        image_org = cv2.imread(file)
        resized = cv2.resize(image_org, (IMAGE_SIZE_YOLOV3, IMAGE_SIZE_YOLOV3), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0

        timer.Timing("preprocess")
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_in})

        timer.Timing("inference")
        boxes = post_processing(img_in, 0.4, 0.6, outputs)

        plot_boxes_cv2(image_org, boxes[0], savename='predictions_onnx.jpg', class_names=CLASS_NAMES)

if __name__ == '__main__':
    main()
