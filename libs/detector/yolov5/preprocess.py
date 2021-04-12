import numpy as np
import torch

def pre_process(inp_img):
    # device = ''
    # cpu = device.lower() == 'cpu'
    # cuda = not cpu and torch.cuda.is_available()
    # device = torch.device('cuda:0' if cuda else 'cpu')
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    # img /= 255.0  # 0 - 255 to 0.0 - 1.0

    img = torch.from_numpy(inp_img)
    img = img.half()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    img = img.permute([0, 3, 1, 2])

    # inp_img = inp_img.astype(np.float32)
    # inp_img /= 255.0
    # if inp_img.shape[-1] == 3:
    #     inp_img = np.expand_dims(inp_img, axis=0)
    #
    # inp_img = inp_img.transpose([0, 3, 2, 1])



    return img.numpy().astype(np.float32)