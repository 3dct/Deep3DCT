from keras.models import *
from keras.layers import Convolution3D, Input, concatenate, RepeatVector, Activation, add, Conv3DTranspose
from keras.layers.advanced_activations import PReLU
from keras.optimizers import *
from keras.engine.topology import Layer
import functools


def vnet3D(pretrained_weights = None,input_size = (128,128,128,1)):
    # Layer 1
    input_layer = Input(shape=input_size, name='data')
    conv_1 = Convolution3D(16, 5, 5, 5, border_mode='same', dim_ordering='tf')(input_layer)
    repeat_1 = concatenate([input_layer] * 16)
    add_1 = add([conv_1, repeat_1])
    prelu_1_1 = PReLU()(add_1)
    downsample_1 = Convolution3D(32, 2,2,2, subsample=(2,2,2))(prelu_1_1)
    prelu_1_2 = PReLU()(downsample_1)

    # Layer 2,3,4
    out2, left2 = downward_layer(prelu_1_2, 2, 64)
    out3, left3 = downward_layer(out2, 2, 128)
    out4, left4 = downward_layer(out3, 2, 256)

    # Layer 5
    conv_5_1 = Convolution3D(256, 5, 5, 4, border_mode='same', dim_ordering='tf')(out4)
    prelu_5_1 = PReLU()(conv_5_1)
    conv_5_2 = Convolution3D(256, 5, 5, 4, border_mode='same', dim_ordering='tf')(prelu_5_1)
    prelu_5_2 = PReLU()(conv_5_2)
    conv_5_3 = Convolution3D(256, 5, 5, 4, border_mode='same', dim_ordering='tf')(prelu_5_2)
    add_5 = add([conv_5_3, out4])
    prelu_5_1 = PReLU()(add_5)
    downsample_5 = Conv3DTranspose(128, (2,2,2), strides=(2,2,2))(prelu_5_1)
    prelu_5_2 = PReLU()(downsample_5)

    #Layer 6,7,8
    out6 = upward_layer(prelu_5_2, left4, 3, 64)
    out7 = upward_layer(out6, left3, 3, 32)
    out8 = upward_layer(out7, left2, 2, 16)

    #Layer 9
    merged_9 = concatenate([out8, add_1], axis=4)
    conv_9_1 = Convolution3D(32, 5, 5, 5, border_mode='same', dim_ordering='tf')(merged_9)
    add_9 = add([conv_9_1, merged_9])
    conv_9_2 = Convolution3D(2, 1, 1, 1, border_mode='same', dim_ordering='tf')(add_9)

    softmax = Convolution3D(1, 1, activation = 'softmax')(conv_9_2)

    model = Model(input_layer, softmax)

    return model

def downward_layer(input_layer, n_convolutions, n_output_channels):
    inl = input_layer
    for _ in range(n_convolutions-1):
        inl = PReLU()(
            Convolution3D(n_output_channels // 2, 5, 5, 5, border_mode='same', dim_ordering='tf')(inl)
        )
    conv = Convolution3D(n_output_channels // 2, 5, 5, 5, border_mode='same', dim_ordering='tf')(inl)
    add1 = add([conv, input_layer])
    downsample = Convolution3D(n_output_channels, 2,2,2, subsample=(2,2,2))(add1)
    prelu = PReLU()(downsample)
    return prelu, add1

def upward_layer(input0 ,input1, n_convolutions, n_output_channels):
    merged = concatenate([input0, input1],  axis=4)
    inl = merged
    for _ in range(n_convolutions-1):
        inl = PReLU()(
            Convolution3D(n_output_channels * 4, 5, 5, 5, border_mode='same', dim_ordering='tf')(inl)
        )
    conv = Convolution3D(n_output_channels * 4, 5, 5, 5, border_mode='same', dim_ordering='tf')(inl)
    add1 = add([conv, merged])
    shape = add1.get_shape().as_list()
    new_shape = (1, shape[1] * 2, shape[2] * 2, shape[3] * 2, n_output_channels)
    upsample =  Conv3DTranspose(n_output_channels, (2,2,2), strides=(2,2,2))(add1)
    return PReLU()(upsample)


