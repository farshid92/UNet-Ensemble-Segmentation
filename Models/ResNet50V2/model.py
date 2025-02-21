import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2))(inputs)
    x = Conv2D(num_filters, 3, padding="same")(x)
    skip_features = tf.image.resize(skip_features, [tf.shape(x)[1], tf.shape(x)[2]])
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50v2_unet(input_shape):
    inputs = tf.keras.Input(input_shape)

    # Load ResNet50V2 as the encoder
    resnet50v2 = ResNet50V2(include_top=False, weights="imagenet", input_tensor=inputs)
    resnet50v2.trainable = False

    # Encoder
    s1 = resnet50v2.get_layer("conv1_conv").output               # (None, 128, 128, 64)
    s2 = resnet50v2.get_layer("conv2_block3_out").output         # (None, 64, 64, 256)
    s3 = resnet50v2.get_layer("conv3_block4_out").output         # (None, 32, 32, 512)
    s4 = resnet50v2.get_layer("conv4_block6_out").output         # (None, 16, 16, 1024)

    # Bridge
    b1 = resnet50v2.get_layer("conv5_block3_out").output         # (None, 8, 8, 2048)

    # Decoder
    d1 = decoder_block(b1, s4, 1024)                             # (None, 16, 16, 1024)
    d2 = decoder_block(d1, s3, 512)                              # (None, 32, 32, 512)
    d3 = decoder_block(d2, s2, 256)                              # (None, 64, 64, 256)
    d4 = decoder_block(d3, s1, 64)                               # (None, 128, 128, 64)
    outputs = UpSampling2D((2, 2))(d4)                           # (None, 256, 256, 64)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs, name="U-Net-ResNet50V2")
    return model
