import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Model


def QNetConv(input_img):
    conv2d_1 = tf.keras.layers.Conv2D(
        120, 60, 3, padding="same", activation="relu", input_shape=(640, 360, 3)
    )(input_img)
    pool_1 = tf.keras.layers.MaxPool2D(pool_size=(65, 25), padding="same")(conv2d_1)
    tf.keras.layers.Dropout(0.5)(pool_1)

    conv2d_2 = tf.keras.layers.Conv2D(60, 30, 3, padding="same", activation="relu")(
        pool_1
    )
    pool_2 = tf.keras.layers.MaxPool2D(pool_size=(65, 25), padding="same")(conv2d_2)
    tf.keras.layers.Dropout(0.5)(pool_2)

    conv2d_3 = tf.keras.layers.Conv2D(60, 25, 3, padding="same", activation="relu")(
        pool_2
    )
    pool_3 = tf.keras.layers.MaxPool2D(pool_size=(65, 25), padding="same")(conv2d_3)
    tf.keras.layers.Dropout(0.5)(pool_3)

    conv2d_4 = tf.keras.layers.Conv2D(120, 60, 3, padding="same", activation="relu")(
        pool_3
    )
    pool_4 = tf.keras.layers.MaxPool2D(pool_size=(65, 25), padding="same")(conv2d_4)
    tf.keras.layers.Dropout(0.5)(pool_4)
    flatten = tf.keras.layers.Flatten()(pool_4)
    return flatten


def PredQValue():
    input_layer = tf.keras.Input(shape=(24, 24), name="input")

    dense = tf.keras.layers.Dense(2, activation="relu")(input_layer)
    # tf.keras.layers.Dropout(2, 0.25)(dense)
    ouput = tf.keras.layers.Softmax()(dense)
    return Model(input_layer, ouput)


PredQValue = PredQValue()


def QNet(x, qvalue, y):
    cnn = np.asarray(QNetConv(x))
    concat = np.append(cnn, qvalue)
    print(concat.shape)

    action = PredQValue(concat, y)
    return action

    # dense = tf.keras.layers.Dense(activation='relu')(concat)
    # tf.keras.layers.Dropout(2, 0.25)(dense)
    # ouput = tf.keras.layers.Softmax()(dense)
