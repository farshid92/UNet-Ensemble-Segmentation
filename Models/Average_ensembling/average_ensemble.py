import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir
np_config.enable_numpy_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

H = 256
W = 256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)                    ## (256, 256)
    return ori_x, x

def save_results(ori_x, ori_y, y_pred, save_image_path):
    # Ensure ori_y and y_pred have the same shape as ori_x
    if ori_x.shape[:2] != ori_y.shape:
        ori_y = cv2.resize(ori_y, (ori_x.shape[1], ori_x.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    if ori_x.shape[:2] != y_pred.shape:
        y_pred = cv2.resize(y_pred, (ori_x.shape[1], ori_x.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert ori_y and y_pred to have the same number of channels as ori_x
    ori_y = np.expand_dims(ori_y, axis=-1)  # (H, W, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)  # (H, W, 3)
        
    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    line = np.ones((H, 10, 3)) * 255
    
    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)
    print(f"Saved result image to: {save_image_path}")
    
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
    
    
    """folder for saving ensembling results"""
    results_dir= "results_average_ensemble"
    create_dir(results_dir)
    
    
    """load the models for average ensembling"""
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model_unet = tf.keras.models.load_model("D:/UNet/files/model508lr-5_rbwTrue.keras")
        model_convnextbase = tf.keras.models.load_model("D:/ConvNextBase/files/model_convnextbase.keras")
        model_mobilenetv3large = tf.keras.models.load_model("D:/MobileNetV3Large/files/model_mobilenetv3large.keras")
        model_resnet50v2 = tf.keras.models.load_model("D:/ResNet50V2/files/model_resnet50v2.keras")
        model_vgg19 = tf.keras.models.load_model("D:/VGG19/files/model_vgg19_unet.keras")
       
    """ Load the test data """
    dataset_path = "D:/thesis/ISIC_Challenge_Dataset"
    _, _, (test_x, test_y) = load_data(dataset_path) 
    
    # Debug: print number of test samples
    print(f"Number of test samples: {len(test_x)}")
    
    
    SCORE = []
    metrics_dict = {'Accuracy': [], 'F1': [], 'Jaccard': [], 'Recall': [], 'Precision': []}  # Initialize metrics_dict

    # for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):

        """ Extracting the image name """
        name = os.path.basename(x)

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)
        ori_h, ori_w = y.shape[:2]
        
        """ Predicting the mask using all models and combining predictions """
        y_pred_unet = model_unet.predict(x)[0]
        y_pred_convnextbase = model_convnextbase.predict(x)[0]
        y_pred_mobilenetv3large = model_mobilenetv3large.predict(x)[0]
        y_pred_resnet50v2 = model_resnet50v2.predict(x)[0]
        y_pred_vgg19 = model_vgg19(x)[0]
        
        # Ensemble strategy: average of predictions
        y_pred = (y_pred_unet+y_pred_convnextbase+y_pred_mobilenetv3large+y_pred_resnet50v2+y_pred_vgg19) / 5.0
        y_pred = (y_pred >= 0.5).numpy().astype(np.float32)  # Convert to binary
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
        df.to_csv("files/score_ensemble.csv")
    
    else:
        print("No scores to display. Please check the dataset and the prediction process.")

