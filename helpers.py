# pylint: skip-file
#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================


from matplotlib import pyplot as plt
import numpy as np
import lpips
import contextlib
import io
import warnings

# 모든 경고를 무시합니다.
warnings.filterwarnings("ignore")


def rgb_to_yuv(img):
    """
    Converts RGB image to YUV

    :param img:
        Input image to perform conversion on
    :return:
        The converted image from `source` to `target`
    """
    rgb_weights = np.array([65.481, 128.553, 24.966])
    img = np.matmul(img, rgb_weights) + 16.

    return img


def compute_psnr(img_pred, img_true, data_range=255., eps=1e-8):
    """
    Compute PSNR between super-resolved and original images.

    :param img_pred:
        The super-resolved image obtained from the model
    :param img_true:
        The original high-res image
    :param data_range:
        Default = 255
    :param eps:
        Default = 1e-8
    :return:
        PSNR value
    """
    err = (img_pred - img_true) ** 2
    err = np.mean(err)

    return 10. * np.log10((data_range ** 2) / (err + eps))


def evaluate_psnr(y_pred, y_true):
    """
    Evaluate individual PSNR metric for each super-res and actual high-res image-pair.

    :param y_pred:
        The super-resolved image from the model
    :param y_true:
        The original high-res image
    :return:
        The evaluated PSNR metric for the image-pair
    """
    if y_pred.shape != y_true.shape:
        y_pred, y_true = center_crop_images(y_pred, y_true)

    y_pred = y_pred.permute((1, 2, 0))  # CHW > HWC
    y_pred = y_pred.cpu().detach().numpy() # torch > numpy
    y_pred = rgb_to_yuv(y_pred)

    y_true = y_true.permute((1, 2, 0))  # CHW > HWC
    y_true = y_true.cpu().detach().numpy() # torch > numpy
    y_true = rgb_to_yuv(y_true)

    psnr = compute_psnr(y_pred, y_true)
    return psnr.item()

import torch
import torch.nn.functional as F

def center_crop_images(y_pred, y_true):
    _, h1, w1 = y_pred.shape  
    _, h2, w2 = y_true.shape  

    crop_height = min(h1, h2)
    crop_width = min(w1, w2)

    startx_pred = w1 // 2 - crop_width // 2
    starty_pred = h1 // 2 - crop_height // 2
    y_pred_cropped = y_pred[:, starty_pred:starty_pred + crop_height, startx_pred:startx_pred + crop_width]

    startx_true = w2 // 2 - crop_width // 2
    starty_true = h2 // 2 - crop_height // 2
    y_true_cropped = y_true[:, starty_true:starty_true + crop_height, startx_true:startx_true + crop_width]

    return y_pred_cropped, y_true_cropped

def evaluate_lpips(y_pred, y_true):
    """
    Evaluate individual PSNR metric for each super-res and actual high-res image-pair.

    :param y_pred:
        The super-resolved image from the model
    :param y_true:
        The original high-res image
    :return:
        The evaluated PSNR metric for the image-pair
    """
    y_pred = y_pred.cpu().detach()
    y_true = y_true.cpu().detach()

    with contextlib.redirect_stdout(io.StringIO()):
        loss_fn_alex = lpips.LPIPS(net='alex')
    lpips_value = loss_fn_alex(y_pred, y_true)

    return lpips_value.item()


def evaluate_average_psnr(sr_images, hr_images):
    """
    Evaluate the avg PSNR metric for all test-set super-res and high-res images.

    :param sr_images:
        The list of super-resolved images obtained from the model for the given test-images
    :param hr_images:
        The list of original high-res test-images
    :return:
        Average PSNR metric for all super-resolved and high-res test-set image-pairs
    """
    psnr = []
    for sr_img, hr_img in zip(sr_images, hr_images):
        psnr.append(evaluate_psnr(sr_img, hr_img))

    average_psnr = np.mean(np.array(psnr))

    return average_psnr

def evaluate_average_lpips(sr_images, hr_images):
    """
    Evaluate the avg LPIPS metric for all test-set super-res and high-res images.

    :param sr_images:
        The list of super-resolved images obtained from the model for the given test-images
    :param hr_images:
        The list of original high-res test-images
    :return:
        Average PSNR metric for all super-resolved and high-res test-set image-pairs
    """
    lpips_lst = []
    for sr_img, hr_img in zip(sr_images, hr_images):
        lpips_lst.append(evaluate_lpips(sr_img, hr_img))

    average_lpips= np.mean(np.array(lpips_lst))

    return average_lpips