import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Large

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2))(inputs)
    x = Conv2D(num_filters, 3, padding="same")(x)
    # Resize skip_features to match the upsampled x
    skip_features = tf.image.resize(skip_features, [tf.shape(x)[1], tf.shape(x)[2]])
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet_mobilenetv3(input_shape):
    inputs = tf.keras.Input(input_shape)

    """ Pre-trained MobileNetV3Large as the encoder """
    mobilenetv3 = MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=inputs)
    mobilenetv3.trainable = False

    """ Encoder """
    s1 = mobilenetv3.get_layer("Conv").output  # (None, 128, 128, 16)
    s2 = mobilenetv3.get_layer("expanded_conv/project/BatchNorm").output  # (None, 64, 64, 24)
    s3 = mobilenetv3.get_layer("expanded_conv_2/project/BatchNorm").output  # (None, 32, 32, 40)
    s4 = mobilenetv3.get_layer("expanded_conv_5/project/BatchNorm").output  # (None, 16, 16, 112)

    """ Bridge """
    b1 = mobilenetv3.get_layer("expanded_conv_10/project/BatchNorm").output  # (None, 8, 8, 160)

    """ Decoder """
    d1 = decoder_block(b1, s4, 112)  # (None, 16, 16, 112)
    d2 = decoder_block(d1, s3, 40)   # (None, 32, 32, 40)
    d3 = decoder_block(d2, s2, 24)   # (None, 64, 64, 24)
    d4 = decoder_block(d3, s1, 16)   # (None, 128, 128, 16)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net-MobileNetV3Large")
    return model

if __name__ == "__main__":
    model = build_unet_mobilenetv3((256, 256, 3))
    model.summary()
