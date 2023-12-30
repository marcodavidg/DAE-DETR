
from __future__ import print_function


import os

import numpy as np
import cv2
import torch
from modelB4 import LDC
IS_LINUX = True
import matplotlib.pyplot as plt


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def get_model():
    checkpoint_path = '/mnt/data0/marco/LDC/checkpoints/BIPED/16/16_model.pth'
    device = 'cuda'

    model = LDC().to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))
    return model

def testLDC(sample, model=None):
    # return
    if model is None:
        # breakpoint()
        checkpoint_path = '/mnt/data0/marco/LDC/checkpoints/BIPED/16/16_model.pth'
        device = 'cuda'

        model = LDC().to(device)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint filte note found: {checkpoint_path}")
        print(f"Restoring weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path,
                                         map_location=device))

    model.eval()
    # breakpoint()
    tensor = model(sample)
    img_shape = tensor[0].shape[2:]
    # breakpoint()
    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
    # print(f"tensor shape: {tensor.shape}")

    image_shape = [x for x in img_shape]
    # (H, W) -> (W, H)
    image_shape = [image_shape[1], image_shape[0]]


    idx = 0
    tmp = tensor[:, idx, ...]
    # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
    tmp = np.squeeze(tmp)

    # Iterate our all 7 NN outputs for a particular image
    preds = []
    # fuse = None
    fuse_num = tmp.shape[0] - 1
    fuse = None
    for i in range(tmp.shape[0]):
        tmp_img = tmp[i]
        tmp_img = np.uint8(image_normalization(tmp_img))
        tmp_img = cv2.bitwise_not(tmp_img)

        # Resize prediction to match input image size
        if not tmp_img.shape[1] == image_shape[0] or not tmp_img.shape[0] == image_shape[1]:
            tmp_img = cv2.resize(tmp_img, (image_shape[0], image_shape[1]))

        preds.append(tmp_img)

        # if i == fuse_num:
        #     # print('fuse num',tmp.shape[0], fuse_num, i)
        #     fuse = tmp_img
        #     fuse = fuse.astype(np.uint8)
    # plt.imshow(normalized_data)
    # plt.savefig('originalImage.png')
    average = np.array(preds, dtype=np.float32)
    average = np.uint8(np.mean(average, axis=0))
    return average

