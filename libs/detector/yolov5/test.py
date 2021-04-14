#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 焦子傲
@contact: 1054568970@qq.com
@file: test.py
@time: 2021/4/9 14:05
@desc:
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../../'))

import argparse

from libs.detector.ssd.onnxmodel import ONNXModel
import time
import cv2
from libs.detector.utils.timer import Timer
from libs.detector.utils.file import Walk
from libs.detector.yolov5.postprocess.postprocess import IMAGE_SIZE_YOLOV5, THRESHOLD_YOLOV5
from libs.detector.yolov5.postprocess.postprocess import PostProcessor_YOLOV5
from libs.detector.yolov5.preprocess import pre_process

import torch

CLASS_NAMES = [
    'background',
    'person',
]
SCORE_ID = 4

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
    net = ONNXModel(args.onnx)
    time_end2 = time.time()
    print('load model cost', time_end2 - time_start1)

    for i, file in enumerate(sorted(image_paths)):
        timer = Timer()
        image_org = cv2.imread(file)

        timer.Timing("read image")
        gray = cv2.resize(image_org, (640, 640))
        img_input = pre_process(gray)
        timer.Timing("preprocess")

        out = net.forward(img_input)
        # pred = model(img, augment=opt.augment)[0]
        timer.Timing("inference")
        print()
        results_batch = PostProcessor_YOLOV5(out)

        for result in results_batch:
            if len(result) > 0:
                result = result.tolist()
                result = [r for r in result if r[SCORE_ID] > THRESHOLD_YOLOV5]  # confidence threshold
                for r in result:
                    x, y, x2, y2, score, label = r
                    if label != 0:
                        continue

                    # y = y / 1.6  # / 640 * 400
                    # y2 = y2 / 1.6  # / 640 * 400
                    x, y, x2, y2, score, label = int(x), int(y), int(x2), int(y2), float(score), int(label)
                    cv2.rectangle(gray, (x, y), (x2, y2), (0, 255, 0))
                    cv2.putText(640, "CLASS_NAME"+" {:.2f}".format(score), (max(0, x), max(15, y+5))
                                ,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    # cv2.putText(img_resize, CLASS_NAMES[int(label)]+" {:.2f}".format(score), (max(0, x), max(15, y+5))
                    #             ,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                image = cv2.resize(gray, (int(gray.shape[1]/gray.shape[0]*960), 960))
                cv2.imshow("draw", image)
                cv2.waitKey(0)

if __name__ == '__main__':
    main()