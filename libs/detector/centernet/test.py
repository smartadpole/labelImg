#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: test.py
@time: 2021/4/6 上午11:53
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
from libs.detector.centernet.postprocess.postprocess import PostProcessor_CENTER_NET
from libs.detector.utils.file import Walk
from libs.detector.centernet.postprocess.postprocess import IMAGE_SIZE_CENTER_NET, CONFIDENCE_THRESHOLD
from libs.detector.centernet.preprocess import pre_process

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
        gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        timer.Timing("read image")
        #
        img_input, meta = pre_process(gray, IMAGE_SIZE_CENTER_NET, 1, None)
        timer.Timing("preprocess")

        out = net.forward(img_input)
        timer.Timing("inference")
        print()
        results_batch = PostProcessor_CENTER_NET(out, meta)

        for result in results_batch.values():
            if len(result) > 0:
                result = result.tolist()
                result = [r for r in result if r[SCORE_ID] > CONFIDENCE_THRESHOLD]
                for r in result:
                    x, y, x2, y2, score = r
                    x, y, x2, y2, score = int(x), int(y), int(x2), int(y2), float(score)
                    #
                    cv2.rectangle(gray, (x, y), (x2, y2), (0, 255, 0))
                    # cv2.putText(img_resize, CLASS_NAMES[int(label)]+" {:.2f}".format(score), (max(0, x), max(15, y+5))
                    #             ,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                image = cv2.resize(gray, (int(gray.shape[1]/gray.shape[0]*960), 960))
                cv2.imshow("draw", image)
                cv2.waitKey(0)

if __name__ == '__main__':
    main()