from typing import Tuple

from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Sequential, Model

LATENT_SHAPE = (1, 1, 100)  # shape of the latent vector


def generator() -> Model:
    """
    Takes a latent vector as input and maps it to data space
    """
    model = Sequential()
    model.add(
        Conv2DTranspose(filters=1024, kernel_size=4, strides=1, use_bias=False, kernel_initializer='glorot_normal',
                        kernel_regularizer='l2', input_shape=LATENT_SHAPE))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False,
                        kernel_initializer='glorot_normal', kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False,
                        kernel_initializer='glorot_normal', kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False,
                        kernel_initializer='glorot_normal', kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', use_bias=False,
                        kernel_initializer='glorot_normal', kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', use_bias=False,
                        kernel_initializer='glorot_normal', kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())

    return model


def discriminator(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    """
    Takes an image as an input and maps it to action space
    """
    model = Sequential()
    model.add(
        Conv2D(filters=64, kernel_size=4, padding='valid', kernel_initializer='glorot_normal', kernel_regularizer='l2',
               use_bias=False, input_shape=input_shape))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        Conv2D(filters=128, kernel_size=4, padding='valid', kernel_initializer='glorot_normal', use_bias=False,
               kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(strides=2))
    model.add(
        Conv2D(filters=256, kernel_size=4, padding='valid', kernel_initializer='glorot_normal', use_bias=False,
               kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        Conv2D(filters=512, kernel_size=4, padding='valid', kernel_initializer='glorot_normal', use_bias=False,
               kernel_regularizer='l2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(strides=2))
    model.add(Conv2D(filters=27, kernel_size=1, padding='valid', kernel_initializer='glorot_normal', use_bias=False,
                     kernel_regularizer='l2'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_initializer='glorot_normal', kernel_regularizer='l2', use_bias=False))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(
        layers.Dense(output_shape[0], kernel_initializer='glorot_normal', kernel_regularizer='l2', activation='sigmoid'))

    return model
