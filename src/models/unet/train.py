import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from common.data import H, W, load_data, create_dir, shuffling, tf_dataset
from common.metrics import dice_coef, iou


# Set memory growth to avoid GPU memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        

        

    


def plot_metrics(history):
    """ Plot the training history """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot accuracy
    axes[1].plot(history.history['dice_coef'], label='Training Dice Coefficient')
    axes[1].plot(history.history['val_dice_coef'], label='Validation Dice Coefficient')
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].legend()
    
    plt.show()

def visualize_predictions(model, dataset, num_images=3):
    """ Visualize predictions with ground truth """
    plt.figure(figsize=(15, 5*num_images))
    for i, (image, mask) in enumerate(dataset.take(num_images)):
        pred_mask = model.predict(image)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        image = image.numpy()[0]
        mask = mask.numpy()[0]
        
        plt.subplot(num_images, 3, i*3 + 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(num_images, 3, i*3 + 2)
        plt.imshow(mask[..., 0], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        plt.subplot(num_images, 3, i*3 + 3)
        plt.imshow(pred_mask[..., 0], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    """seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """folder for saving data"""
    create_dir("files")
    
    """Hyperparameters"""
    batch_size = 8
    lr = 1e-5
    num_epoch = 50
    model_path = "files/model508lr-5_rbwTrue.keras" # means 50 epochs, 8 batchsize, learning rate 1e-5, restore_best_weights=True
    csv_path = "files/data508lr-5_rbwTrue.csv"
    log_dir = "files/logs508lr-5_rbwTrue"
    
    """Dataset : 60/20/20"""
    dataset_path = os.environ.get("ISIC_DATASET_PATH", "data/ISIC_Challenge_Dataset")
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")    
    
    
    train_dataset = tf_dataset(train_x,train_y,batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)
    
    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size
    
    if len(train_x)%batch_size !=0:
        train_steps += 1
        
    if len(valid_x)%batch_size !=0:
        valid_steps += 1 
        
        """ Model """
    model = build_unet((H, W, 3))
    metrics = [dice_coef, 
               iou, 
               Recall(thresholds = 0.5), 
               Precision(thresholds = 0.5)]
    model.compile(loss="binary_crossentropy",# dice_loss 
                  optimizer=Adam(learning_rate =lr, clipnorm=1.0), 
                  metrics=metrics)
    model.summary()

    callbacks = [
       ModelCheckpoint(model_path, verbose=1, save_best_only=True),
       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
       CSVLogger(csv_path),
       TensorBoard(log_dir=log_dir),
       EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
   ]
    
    for data, label in train_dataset.take(1):
        print(f"Data shape: {data.shape}, Label shape: {label.shape}")
        
 
    try:
       history = model.fit(
           train_dataset,
           epochs=num_epoch,
           validation_data=valid_dataset,
           steps_per_epoch=train_steps,
           validation_steps=valid_steps,
           callbacks=callbacks
           )
    except Exception as e:
        print(f"Error during model fitting: {e}")
        raise e

    # Plot training history
    plot_metrics(history)
    
    # Visualize predictions
    visualize_predictions(model, valid_dataset)
    