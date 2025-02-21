import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_vgg19_unet
from metrics import dice_coef, iou

# Set memory growth to avoid GPU memory errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path, split=0.2):
    train_images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Training_Input", "*.jpg")))
    train_masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Training_GroundTruth", "*.png")))

    valid_images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Validation_Input", "*.jpg")))
    valid_masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Validation_GroundTruth", "*.png")))

    test_images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Test_Input", "*.jpg")))
    test_masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Test_GroundTruth", "*.png")))

    return (train_images, train_masks), (valid_images, valid_masks), (test_images, test_masks)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # (H, W, 3)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x  # (256, 256, 3)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H, W)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)  # (256, 256)
    x = np.expand_dims(x, axis=-1)  # (256, 256, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

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

    # Plot dice coefficient
    axes[1].plot(history.history['dice_coef'], label='Training Dice Coefficient')
    axes[1].plot(history.history['val_dice_coef'], label='Validation Dice Coefficient')
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].legend()

    plt.show()

def visualize_predictions(model, dataset, num_images=3):
    """ Visualize predictions with ground truth """
    plt.figure(figsize=(15, 5 * num_images))
    for i, (image, mask) in enumerate(dataset.take(num_images)):
        pred_mask = model.predict(image)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        image = image.numpy()[0]
        mask = mask.numpy()[0]

        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(mask[..., 0], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(num_images, 3, i * 3 + 3)
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
    model_path = "files/model_vgg19_unet.keras"
    csv_path = "files/data_vgg19_unet.csv"
    log_dir = "files/logs_vgg19_unet"

    """Dataset : 60/20/20"""
    dataset_path = "your path to ISIC_Challenge_Dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)

    train_steps = len(train_x) // batch_size
    valid_steps = len(valid_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    """ Model """
    model = build_vgg19_unet((H, W, 3))
    metrics = [
        dice_coef,
        iou,
        Recall(thresholds=0.5),
        Precision(thresholds=0.5)
    ]
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        metrics=metrics
    )
    model.summary()

    # Check output shape
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    dummy_output = model.predict(dummy_input)
    print(f"Model output shape: {dummy_output.shape}")

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
