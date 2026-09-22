"""Segmentation metrics shared by every model and ensemble script."""

import tensorflow as tf
from tensorflow.keras import backend as K

smooth = 1e-15


def iou(y_true, y_pred):
    """Intersection over Union (Jaccard index)."""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_coef(y_true, y_pred):
    """Dice coefficient (F1 score over pixels)."""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    """1 - Dice coefficient, usable as a Keras loss."""
    return 1.0 - dice_coef(y_true, y_pred)
