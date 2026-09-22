import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from common.metrics import dice_loss, dice_coef, iou
from common.data import load_data, create_dir
from common.inference import H, W, read_image, read_mask, save_results


    
def plot_metrics(metrics, names, output_dir):
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(12, 8))
        plt.plot(metric, label=names[i])
        plt.xlabel('Samples')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid()
        plt.title(f'{names[i]} Metric')
        plt.savefig(os.path.join(output_dir, f'{names[i]}_metric.png'))
        plt.close()

if __name__ == "__main__":
    """seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """folder for saving results"""
    results_dir= "results_resnet50v2"
    create_dir(results_dir)
    
    """load the model"""
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model_resnet50v2.keras")
       
    """ Load the test data """
    dataset_path = os.environ.get("ISIC_DATASET_PATH", "data/ISIC_Challenge_Dataset")
    _, _, (test_x, test_y) = load_data(dataset_path) 
    
    # Debug: print number of test samples
    print(f"Number of test samples: {len(test_x)}")
    
    SCORE = []
    metrics_dict = {'Accuracy': [], 'F1': [], 'Jaccard': [], 'Recall': [], 'Precision': []}  # Initialize metrics_dict

    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting the image name """
        name = os.path.basename(x)

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)
        ori_h, ori_w = y.shape[:2]
        
      
        """ Predicting the mask """
        y_pred = model.predict(x)[0]  # Get the raw prediction
        y_pred = (y_pred >= 0.5).astype(np.float32)  # Convert to binary
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = cv2.resize(y_pred, (ori_w, ori_h))
        
        """ Saving the predicted mask """
        save_image_path = os.path.join(results_dir, name)
        save_results(ori_x, ori_y, y_pred, save_image_path)
    
        """ Flatten the array """
        y = (y > 0).astype(np.float32)  # Ensure binary format
        y_pred = (y_pred > 0).astype(np.float32)
        y = y.flatten()
        y_pred = y_pred.flatten()
        
        # Check shapes and types
        print(f"Shape of y: {y.shape}, type: {y.dtype}")
        print(f"Shape of y_pred: {y_pred.shape}, type: {y_pred.dtype}")

        """ Print unique values for debugging """
        print(f"Unique values in y: {np.unique(y)}")
        print(f"Unique values in y_pred: {np.unique(y_pred)}")
        
        # Ensure binary format
        y = (y > 0).astype(np.float32)
        y_pred = (y_pred > 0).astype(np.float32)
        
        # Ensure there are no NaNs or Infs
        assert not np.isnan(y).any() and not np.isinf(y).any(), "NaNs or Infs detected in y"
        assert not np.isnan(y_pred).any() and not np.isinf(y_pred).any(), "NaNs or Infs detected in y_pred"

        """ Calculating metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])
        
        metrics_dict['Accuracy'].append(acc_value)
        metrics_dict['F1'].append(f1_value)
        metrics_dict['Jaccard'].append(jac_value)
        metrics_dict['Recall'].append(recall_value)
        metrics_dict['Precision'].append(precision_value)

    """ Mean metrics values """
    if SCORE:
        score = [s[1:] for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1: {score[1]:0.5f}")
        print(f"Jaccard: {score[2]:0.5f}")
        print(f"Recall: {score[3]:0.5f}")
        print(f"Precision: {score[4]:0.5f}")
    
        df = pd.DataFrame(SCORE, columns=["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
        df.to_csv("files/score508lr-5_rbwTrue.csv")
    
        # Plot metrics
        metric_names = list(metrics_dict.keys())
        metrics = [metrics_dict[name] for name in metric_names]
        # plot_metrics(metrics, metric_names, "files")
    else:
        print("No scores to display. Please check the dataset and the prediction process.")