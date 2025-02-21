from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,Activation, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = concatenate([x, skip_features], axis=-1)
    x = conv_block(x, num_filters)
    return x

def build_vgg19_unet(input_shape):
    inputs = Input(input_shape)
    
    # Load VGG19 without top layers
    vgg19_base = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Freeze the VGG19 layers
    for layer in vgg19_base.layers:
        layer.trainable = False
    
    # Encoder blocks from VGG19
    s1 = vgg19_base.get_layer('block1_conv2').output
    s2 = vgg19_base.get_layer('block2_conv2').output
    s3 = vgg19_base.get_layer('block3_conv4').output
    s4 = vgg19_base.get_layer('block4_conv4').output
    
    b1 = conv_block(vgg19_base.get_layer('block5_conv4').output, 1024)
    
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    
    model = Model(inputs, outputs)
    return model
