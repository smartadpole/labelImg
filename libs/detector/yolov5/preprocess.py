import numpy as np
import cv2
from .postprocess.postprocess import IMAGE_SIZE_YOLOV5

def pre_process(inp_img):

    img = cv2.resize(inp_img, (IMAGE_SIZE_YOLOV5, IMAGE_SIZE_YOLOV5))
    img = np.half(img)  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.shape[-1] == 3:
        img = np.expand_dims(img, 0)

    img = np.transpose(img, [0, 3, 1, 2])

    return img.astype(np.float32)