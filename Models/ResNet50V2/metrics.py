# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras import backend as K

smooth = 1e-15

def iou(y_true, y_pred):
    """Calculate Intersection over Union (IoU)"""
    """add line here"""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def dice_coef(y_true, y_pred):
    """Calculate Dice Coefficient"""
    """add line here"""
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    """Calculate Dice Loss"""
    return 1.0 - dice_coef(y_true, y_pred)