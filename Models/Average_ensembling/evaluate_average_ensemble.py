import numpy as np
import os
import cv2
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from train import load_data, create_dir
from softprediction import load_soft_predictions
import tensorflow as tf

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Using GPU: {gpus}")
else:
    print("No GPU found, using CPU.")

# Load best weights
best_weights = np.load('best_weights.npy')

# Load the validation data
dataset_path = "D:/thesis/ISIC_Challenge_Dataset"
_, (valid_x, valid_y), _ = load_data(dataset_path)

# Directory where the soft predictions for validation data are stored
soft_predictions_dir = "results/seperated_soft_predictions"

# Metrics to calculate
iou_scores = []
dice_scores = []
precision_scores = []
recall_scores = []

# Evaluate on validation data (100 images)
for i, y_path in enumerate(valid_y):
    try:
        # Load the ground truth mask
        y_true = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        y_true = cv2.resize(y_true, (256, 256))
        y_true = (y_true > 0.5).astype(np.float32).flatten()

        # Load the soft predictions
        soft_preds = load_soft_predictions(soft_predictions_dir, i)

        # Calculate the weighted average using the best weights
        weighted_pred = np.tensordot(best_weights, soft_preds, axes=(0, 0)).flatten()

        # Threshold the weighted prediction
        weighted_pred = (weighted_pred > 0.5).astype(np.float32)

        # Calculate metrics
        iou_scores.append(jaccard_score(y_true, weighted_pred))
        dice_scores.append(f1_score(y_true, weighted_pred))
        precision_scores.append(precision_score(y_true, weighted_pred))
        recall_scores.append(recall_score(y_true, weighted_pred))
    except FileNotFoundError:
        print(f"File for index {i} does not exist. Skipping...")

# Compute average metrics
average_iou = np.mean(iou_scores)
average_dice = np.mean(dice_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)

# Print the results
print(f"Average IOU: {average_iou:.4f}")
print(f"Average Dice Coefficient: {average_dice:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
