#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: test.py
@time: 2021/2/2 下午5:49
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../../'))

import argparse

from libs.detector.ssd.onnxmodel import ONNXModel
import time
import cv2
import  numpy as np
from libs.detector.utils.timer import Timer
from libs.detector.ssd.postprocess.ssd import PostProcessor_SSD
from libs.detector.utils.file import Walk
from libs.detector.ssd.postprocess.ssd import IMAGE_SIZE_SSD

PIXEL_MEAN = [123, 117, 104]
THRESHOLD = [1, 1, 1, 1, 0.195, 1, 1, 0.353, 1, 1, 1]
CLASS_NAMES = [
    'background',
    'pet-cat', 'pet-dog', 'excrement', 'wire', 'key',
    'weighing-scale', 'shoes', 'socks', 'power-strip', 'base',
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
    net = ONNXModel(args.onnx)
    time_end2 = time.time()
    print('load model cost', time_end2 - time_start1)

    for i, file in enumerate(sorted(image_paths)):
        timer = Timer()
        image_org = cv2.imread(file)
        gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        timer.Timing("read image")
        img_resize = cv2.resize(gray, (IMAGE_SIZE_SSD, IMAGE_SIZE_SSD), interpolation = cv2.INTER_CUBIC)
        # img_input = img_input[..., ::-1] # BGR to RGB
        img_input = (img_resize - PIXEL_MEAN).astype(np.float32).transpose((2, 0, 1))[np.newaxis, ::]
        timer.Timing("preprocess")

        out = net.forward(img_input)
        timer.Timing("inference")
        print()
        results_batch = PostProcessor_SSD(out[0], out[1], out[2])

        for result in results_batch:
            if len(result) > 0:
                result = result.tolist()
                for r in result:
                    x, y, x2, y2, label, score = r
                    x, y, x2, y2, label, score = int(x), int(y), int(x2), int(y2), int(label), float(score)

                    if score < THRESHOLD[label]:
                        continue
                    cv2.rectangle(img_resize, (x, y), (x2, y2), (0, 255, 0))
                    cv2.putText(img_resize, CLASS_NAMES[int(label)]+" {:.2f}".format(score), (max(0, x), max(15, y+5))
                                ,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                image = cv2.resize(img_resize, (int(img_resize.shape[1]/img_resize.shape[0]*960), 960))
                cv2.imshow("draw", image)
                cv2.waitKey(0)

        # print(labels[np.argmax(out[0][0])])
        # cv2.putText(img_ori, labels[np.argmax(out[0][0])], (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
        # cv2.imshow("1", img_ori)
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()