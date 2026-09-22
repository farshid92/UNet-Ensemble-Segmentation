"""Loading of per-model soft predictions used by the ensembling scripts.

Each trained model exports a per-pixel probability map per image as
``<model>_preds_<index>.npy``. The ensembling code blends these maps rather
than the binarised masks, so the optimiser can work with continuous values.
"""

import os

import cv2
import numpy as np

MODELS = ["unet", "convnextbase", "mobilenetv3large", "resnet50v2", "vgg19"]

SOFT_PREDICTIONS_DIR = os.environ.get("SOFT_PREDICTIONS_DIR", "results/soft_predictions")


def load_soft_predictions(soft_predictions_dir, index):
    """Return the five models' soft predictions for one image, stacked."""
    return np.array([
        np.load(os.path.join(soft_predictions_dir, f"{model}_preds_{index}.npy"))
        for model in MODELS
    ])


def load_and_process_mask(mask_path):
    """Read a ground truth mask, resize to 256x256 and scale to [0, 1]."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    return mask / 255.0
