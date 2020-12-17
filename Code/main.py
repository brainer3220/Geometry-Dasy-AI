import datetime
import glob
import os
import time

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
Epoch = 1001  # Game repeat count
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

SEED = 2020

# Funciton


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
    if col_type != object:
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == "int":
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df[col] = df[col].astype(np.uint8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df[col] = df[col].astype(np.uint32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
            elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                df[col] = df[col].astype(np.uint64)
        elif str(col_type)[:5] == "float":
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


def Jump():
    """
    Jump Function
    """
    pag.press("space")


def Q_Value(State, Action):
    """
    Q Value Function
    """
    reword + (Discount * max(Q_next))
    # return


def Click_Start():
    """
    Click to Start Button
    """
    pag.moveTo(417, 257)  # X and y coordinates of the start button
    pag.mouseDown()
    pag.mouseUp()


def bring_window():
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
        bring_window()
        print(position_str)


# Full resolution of the emulator
Game_Scr_pos = {"left": 16, "top": 54, "height": 483, "width": 789}

# Where to click the button on the emulator.
Game_Src_Click_pos = [379, 283]


def Real_Time():
    bring_window()
    i = 0
    while True:
        i += 1
        with mss.mss() as sct:
            Game_Scr = np.array(sct.grab(Game_Scr_pos))[:, :, :3]

            # Below is a test to see if you are capturing the screen of the emulator.
            # cv2.imshow('Game_Src', Game_Scr)
            # cv2.waitKey(0)

            Game_Scr = cv2.resize(
                Game_Scr, dsize=(960, 540), interpolation=cv2.INTER_AREA
            )
            # Game_Scr = np.ravel(Game_Scr)

            GMD_Model = os.path.join(os.getcwd(), "Model", "CNN", "saved_model.pb")
            GMD_Model_Keras = os.path.join(
                os.getcwd(), "..", "Model", "Keras", "keras_model.h5"
            )

            # model = saver.restore(sess, GMD_Model)
            data = np.ndarray(shape=(1, 960, 540, 3), dtype=np.float32)

            # Replace this with the path to your image
            image = Image.open(Game_Scr)

            # Make sure to resize all images to 224, 224 otherwise they won't fit in the array
            image = image.resize((960, 540))
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = image_array / 255.0

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)
            print(prediction)

            with tf.Session() as sess:
                graph = tf.Graph()
                with graph.as_default():
                    with tf.name_scope("Convolution"):
                        Gmd = Convolution(Game_Scr)
                    with tf.name_scope("Relu_Function"):
                        Gmd = tf.nn.relu(Gmd)
                    with tf.name_scope("MaxPool"):
                        Gmd = Max_Pool(Gmd)
                    if i == 1:
                        writer = tf.summary.FileWriter(
                            "..\Graph\GMDmiss", graph=tf.get_default_graph()
                        )
                        writer.close()
            print(Gmd.shape)
            print(Gmd)
            # cv2.imshow('Game_Src', Game_Scr)
            # cv2.waitKey(0)

            # CNN
            # model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
            # model.add(Conv2D(64, (3, 3), activation='relu'))


def Video_Analyze(Video):
    Vidcap = cv2.VideoCapture(Video)
    success, image = Vidcap.read()
    count = 0
    while success:
        # save frame as JPEG file
        cv2.imwrite("frame%d.jpg" % count, image)
        success, image = Vidcap.read()
        print("Read a new frame: ", success)
        count += 1


def Game_Play_With_Learning():
    Num_Of_Play_Time = int(input("Press number from Game Time."))
    while True:
        Real_Time()
        Play_Time = time.time()  # Game start time
        Jump()

        # if:        # if ended from One of game, up to Play Time.
        #
        #     Num_Of_Play_Time += 1
        #     Play_Time = time.time() - Play_Time # Playtime for one game

        if epoch > Play_Time:
            break


# def Play_Learning:


def Game_play():
    np.set_printoptions(suppress=True)

    model = load_model("../Model/Keras/keras_model.h5", custom_objects=None)

    while True:
        with mss.mss() as sct:
            Game_Scr = np.array(sct.grab(Game_Scr_pos))[:, :, :3]

            # Below is a test to see if you are capturing the screen of the emulator.
            # cv2.imshow('Game_Src', Game_Scr)
            # cv2.waitKey(0)

            Game_Scr = cv2.resize(
                Game_Scr, dsize=(960, 540), interpolation=cv2.INTER_AREA
            )
            x = np.array(Game_Scr).reshape(-1, 1)

            size = (224, 224)
            image = ImageOps.fit(Game_Scr, size, Image.ANTIALIAS)

            Result = []
            Result = model.predict(x)
            if Result == 0:
                print("Play")
            else:
                print("Miss")


def BinaryImageClassf():
    model = Sequential()
    model.add(
        Conv2D(120, 60, 3, padding="same", activation="relu", input_shape=(640, 360, 3))
    )
    model.add(MaxPooling2D(pool_size=(65, 25)))
    model.add(Dropout(0.25))

    model.add(Conv2D(60, 30, 3, padding="same"))
    model.add(MaxPooling2D(pool_size=(60, 25), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(60, 25, 3, padding="same"))
    model.add(MaxPooling2D(pool_size=(60, 25), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="sigmoid"))

    model.add(Dense(1, activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    First_State = int(
        input(
            """If you want to analyze your video?
press 1.

or real time play game and real time screen analyze.
press 2.

If learning Geometry Dash 'Play Game' and 'Nothing' image
Press 3.

If you gaming from real time
Press 4
"""
        )
    )

    if First_State == 1:
        Video = input("Please enter a video path and video name.")
        Video_Analyze(Video)

    elif First_State == 2:
        Real_Time()

    elif First_State == 4:
        Game_play()

    elif First_State == 3:
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "Photo\\isPlay",
            validation_split=0.2,
            subset="training",
            shuffle=True,
            seed=SEED,
            label_mode="binary",
            image_size=(640, 360),
        )
        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "Photo\\isPlay",
            validation_split=0.2,
            subset="validation",
            shuffle=True,
            seed=SEED,
            label_mode="binary",
            image_size=(640, 360),
        )
        # train_dataset = train_dataset.cache().shuffle(30).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print("Load Dataset")

        # print(train_dataset.class_names)
        print(train_dataset)

        # cv2.imshow('Game_Src', cv2.imread(train_dataset.take(1)))
        # cv2.waitKey(0)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        bin_img_clssf = BinaryImageClassf()
        history = bin_img_clssf.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=30,
            batch_size=2,
            callbacks=[tensorboard_callback],
        )

        model.save("model.h5")
