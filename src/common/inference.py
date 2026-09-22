"""Helpers for evaluation and ensembling.

These differ from common.data: they keep the original image alongside the
normalised tensor so predictions can be saved next to their inputs.
"""

import cv2
import numpy as np

H = 256
W = 256


def read_image(path):
    """Return (original resized image, batched normalised tensor)."""
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.float32)
    return ori_x, np.expand_dims(x, axis=0)


def read_mask(path):
    """Return (original mask, normalised mask)."""
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ori_x = x
    x = x / 255.0
    return ori_x, x.astype(np.float32)


def save_results(ori_x, ori_y, y_pred, save_image_path):
    """Write a side-by-side [input | ground truth | prediction] strip."""
    if ori_x.shape[:2] != ori_y.shape:
        ori_y = cv2.resize(ori_y, (ori_x.shape[1], ori_x.shape[0]), interpolation=cv2.INTER_NEAREST)
    if ori_x.shape[:2] != y_pred.shape:
        y_pred = cv2.resize(y_pred, (ori_x.shape[1], ori_x.shape[0]), interpolation=cv2.INTER_NEAREST)

    ori_y = np.concatenate([np.expand_dims(ori_y, axis=-1)] * 3, axis=-1)
    y_pred = np.concatenate([np.expand_dims(y_pred, axis=-1)] * 3, axis=-1)

    line = np.ones((ori_x.shape[0], 10, 3)) * 255
    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred * 255], axis=1)
    cv2.imwrite(save_image_path, cat_images)
