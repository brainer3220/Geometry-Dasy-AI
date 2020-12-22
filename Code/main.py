import datetime
import glob
import os
import time
import datetime
import random
import shutil

import cv2
import mss
import numpy as np
import pandas as pd
import pyautogui as pag
import tensorflow as tf
from PIL import Image
from PIL import ImageGrab
from PIL import ImageOps
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array

np.set_printoptions(suppress=True)

Epsilon = 1  # Random probability
Epsilon_Minimum_Value = 0.001  # epsilon의 최소값
nbActions = 2  # Number of actions (jump, wait)
EPOCH = 1001  # Game repeat count
Hidden_Size = 100  # Hidden layer count
Max_Memory = 5000  # Maximum number of game contents remembered
batch_Size = 50  # Number of data bundles in training
Grid_Size = 10  # Grid size
nb_States = Grid_Size * Grid_Size  # State count
Discount = 0.9  # discount Value
Learning_Rate = 0.2  # Learning_Rate

Reword_List = []

Replay_Meomry = 100000

reword = 0

RANDOM_STATE = 2020

tf.random.set_seed(RANDOM_STATE)

# Funciton


def Jump():
    """
    Jump Function
    """
    pag.moveTo(416, 275)
    pag.mouseDown()
    time.sleep(0.05)
    pag.mouseUp()


def Retry():
    """
    Retry Funtion
    """
    pag.moveTo(240, 480)
    pag.mouseDown()
    pag.mouseUp()


def Q_Value(State, Action):
    """
    Q Value Function
    """
    reword + (Discount * max(Q_next))
    # return


def BringWindow():
    """
    Bring the emulator to the front
    """
    time.sleep(0.5)
    apple = """
    tell application "BlueStacks"
    activate
    end tell
    """


def average_hash(fname, size=16):
    img = Image.open(fname)
    img = img.convert("L")
    img = img.resize((960, 540), Image.ANTIALIAS)
    pixel_data = img.getdata()
    pixels = np.array(pixel_data)
    pixels = pixels.reshape((960, 540))
    avg = pixels.mean()
    diff = 1 * (pixels > avg)
    print(diff)


def GetResolution():
    """
    Function to get resolution.
    Test it when you bring the emulator's resolution coordinates.
    """
    while True:
        x, y = pag.position()
        position_str = "X: " + str(x) + "Y: " + str(y)
        BringWindow()
        print(position_str)


# Full resolution of the emulator
Game_Scr_pos = {"left": 16, "top": 54, "height": 483, "width": 789}

# Where to click the button on the emulator.
Game_Src_Click_pos = [379, 283]


def VideoAnalyze(Video):
    Vidcap = cv2.VideoCapture(Video)
    success, image = Vidcap.read()
    count = 0
    while success:
        # save frame as JPEG file
        cv2.imwrite("frame%d.jpg" % count, image)
        success, image = Vidcap.read()
        print("Read a new frame: ", success)
        count += 1


def GamePlayWithLearning():




def GamePlay():
    np.set_printoptions(suppress=True)

    model = load_model("../Model/Keras/keras_model.h5", custom_objects=None)

    while True:
        with mss.mss() as sct:
            Game_Scr = np.array(sct.grab(Game_Scr_pos))[:, :, :3]

            # Below is a test to see if you are capturing the screen of the emulator.
            # cv2.imshow('Game_Src', Game_Scr)
            # cv2.waitKey(0)

            Game_Scr = cv2.resize(Game_Scr,
                                  dsize=(960, 540),
                                  interpolation=cv2.INTER_AREA)
            x = np.array(Game_Scr).reshape(-1, 1)

            size = (224, 224)
            image = ImageOps.fit(Game_Scr, size, Image.ANTIALIAS)

            Result = []
            Result = model.predict(x)
            if Result == 0:
                print("Play")
            else:
                print("Miss")


def ImageClassf():
    model = Sequential()
    model.add(Conv2D(120, 60, 3, padding='same', activation='relu',
                        input_shape=(640, 360, 3)))
    model.add(MaxPooling2D(pool_size=(65, 25), padding='same'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(60, 30, 3, padding='same'))
    model.add(MaxPooling2D(pool_size=(60, 25), padding='same'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(60, 25, 3, padding='same'))
    model.add(MaxPooling2D(pool_size=(60, 25), padding='same'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    First_State = int(
        input("""If you want to analyze your video?
press 1.

or real time play game and real time screen analyze.
press 2.

If learning Geometry Dash 'Play Game' and 'Nothing' image
Press 3.

If you gaming from real time
Press 4
"""))

    if First_State == 1:
        Video = input("Please enter a video path and video name.")
        VideoAnalyze(Video)

    elif First_State == 2:
        GamePlayWithLearning()

    elif First_State == 4:
        GamePlay()

    elif First_State == 3:
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "Photo\\isPlay",
            validation_split=0.2,
            subset="training",
            shuffle=True,
            seed=RANDOM_STATE,
            label_mode="categorical",
            image_size=(640, 360),
        )
        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "Photo\\isPlay",
            validation_split=0.2,
            subset="validation",
            shuffle=True,
            seed=RANDOM_STATE,
            label_mode="categorical",
            image_size=(640, 360),
        )
        # train_dataset = train_dataset.cache().shuffle(30).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print("Load Dataset")

        # print(train_dataset.class_names)
        print(train_dataset)

        # cv2.imshow('Game_Src', cv2.imread(train_dataset.take(1)))
        # cv2.waitKey(1)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)

        try:
            bin_img_clssf = load_model("Model\\" +
                                       str(os.listdir("Model")[-1]))
            # bin_img_clssf = ImageClassf()
            print("Model load 성공")
        except:
            bin_img_clssf = ImageClassf()
            print("Model load 실패")

        history = bin_img_clssf.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=2,
            batch_size=64,
            callbacks=[tensorboard_callback],
        )

        bin_img_clssf.save("Model\\" +
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +
                           "model.h5")
