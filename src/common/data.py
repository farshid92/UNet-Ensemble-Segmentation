"""ISIC 2018 data loading and the tf.data training pipeline.

Dataset location is read from the ISIC_DATASET_PATH environment variable;
see the README for the expected directory layout.
"""

import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

H = 256
W = 256

DATASET_PATH = os.environ.get("ISIC_DATASET_PATH", "data/ISIC_Challenge_Dataset")


def create_dir(path):
    """Create a directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x, y):
    """Shuffle images and masks together with a fixed seed."""
    return shuffle(x, y, random_state=42)


def load_data(dataset_path=None):
    """Return (train, valid, test) pairs of image/mask path lists.

    Uses the official ISIC 2018 Task 1 splits: 2594 training, 100 validation
    and 1000 test images, each with an expert-annotated ground truth mask.
    """
    dataset_path = dataset_path or DATASET_PATH

    train_images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Training_Input", "*.jpg")))
    train_masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Training_GroundTruth", "*.png")))

    valid_images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Validation_Input", "*.jpg")))
    valid_masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Validation_GroundTruth", "*.png")))

    test_images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Test_Input", "*.jpg")))
    test_masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Test_GroundTruth", "*.png")))

    return (train_images, train_masks), (valid_images, valid_masks), (test_images, test_masks)


def read_image(path):
    """Read an RGB image, resize to 256x256 and scale to [0, 1]."""
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    return x.astype(np.float32)                      # (256, 256, 3)


def read_mask(path):
    """Read a grayscale mask, resize to 256x256 and scale to [0, 1]."""
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return np.expand_dims(x, axis=-1)                # (256, 256, 1)


def tf_parse(x, y):
    """Wrap read_image/read_mask for use inside a tf.data pipeline."""
    def _parse(x, y):
        return read_image(x), read_mask(y)

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch):
    """Build a batched, prefetched, repeating tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset.repeat()
