import datetime
import glob
import os
import random
import shutil
import time

import cv2
import mss
import numpy as np
import pandas as pd
import tensorflow as tf
from ImageClassf import ImageClassf
from PIL import Image
from PIL import ImageGrab
from PIL import ImageOps
from play import *
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model

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


# Full resolution of the emulator
Game_Scr_pos = {"left": 16, "top": 54, "height": 483, "width": 789}

# Where to click the button on the emulator.
Game_Src_Click_pos = [379, 283]

def ImportImageDataSet():
    return tf.keras.preprocessing.image_dataset_from_directory(
            "Photo\\isPlay",
            validation_split=0.2,
            subset="training",
            shuffle=True,
            seed=RANDOM_STATE,
            label_mode="categorical",
            image_size=(640, 360),
        ), tf.keras.preprocessing.image_dataset_from_directory(
            "Photo\\isPlay",
            validation_split=0.2,
            subset="validation",
            shuffle=True,
            seed=RANDOM_STATE,
            label_mode="categorical",
            image_size=(640, 360),
        )


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


def PlayWithLearning():
    BringWindow()
    # load_model('Model\\20201218-003432model.h5')
    isGamePlay = load_model("Model\\" + str(os.listdir("Model")[-1]))
    print("Model\\" + str(os.listdir("Model")[-1]))
    # last_select = []
    isStart = 0

    for a_epoch in range(EPOCH):
        with mss.mss() as sct:
            Game_Scr = np.array(sct.grab(Game_Scr_pos))[:, :, :3]
            """Below is a test to see if you are capturing the screen of the emulator."""
            # cv2.imshow('Game_Src', Game_Scr)
            # cv2.waitKey(1)

            Game_Scr_numpy = np.resize(Game_Scr, (1, 640, 360, 3))


            if ((tf.math.argmax(isGamePlay.predict(Game_Scr_numpy),
                                       axis=1) == 1) == True) is True:
                rnd = random.randint(1, 10)
                if isStart < 1:
                    if not os.path.exists("tmp"):
                        # os.makedirs('tmp')
                        os.makedirs("tmp\\stay")
                        os.makedirs("tmp\\up")
                    try:
                        dqn = load_model("Model\\Play\\game_play.h5")
                        # dqn = ImageClassf
                        is_load_model = True
                        print("Model load 성공")
                    except:
                        # dqn = Q_net.QNet()
                        is_load_model = False
                        print("Model load 실패")
                    play_time = time.time()
                    print("Play...")

                isStart += 1

                if is_load_model is True:
                    if rnd in [1, 2]:
                        save_path = "stay"
                        print("RAND Stay")
                    elif rnd in [3, 4]:
                        save_path = "up"
                        Jump()
                        print("RAND Up")
                    else:
                        tmp = tf.math.argmax(dqn.predict(Game_Scr_numpy),
                                             axis=1)

                        if tmp == 1:
                            save_path = "stay"
                            print("Stay")
                        else:
                            save_path = "up"
                            Jump()
                            print("Up")
                else:
                    if rnd < 6:
                        save_path = "stay"
                        print("stay")
                    elif rnd >= 6:
                        save_path = "up"
                        Jump()
                        print("up")
                    else:
                        print("It's a problem")
                cv2.imwrite(f"tmp\\{save_path}\\{int(time.time())}.png",
                            Game_Scr)

            elif ((tf.math.argmax(isGamePlay.predict(Game_Scr_numpy),
                                       axis=1) == 1) == True) == False and isStart < 1:
                print("Go!")

            elif ((tf.math.argmax(isGamePlay.predict(Game_Scr_numpy),
                                       axis=1) == 1) == True) == False and isStart > 1:
                play_time = time.time() - play_time
                print("What are you doing?")

                # try:
                # for - in range(2):
                #     print((os.listdir('tmp\\stay') + os.listdir('tmp\\up')).sort()[-1])
                #     os.remove('tmp\\up\\' + os.listdir('tmp\\stay') + os.listdir('tmp\\up').sort(reverse=True)[-1])
                #     os.remove('tmp\\stay\\' + os.listdir('tmp\\stay') + os.listdir('tmp\\up').sort(reverse=True)[-1])
                # os.remove('tmp\\stay\\' + os.listdir('tmp\\stay')[-1])
                # os.remove('tmp\\up\\' + os.listdir('tmp\\up')[-1])
                # for i in range(1):
                #     if save_path == 'stay':
                #         os.remove('tmp\\stay\\' + os.listdir('tmp\\stay')[-1])
                #     elif save_path == 'up':
                #         os.remove('tmp\\up\\' + os.listdir('tmp\\up')[-1])
                # except:
                #     pass

                # try:
                game_play = tf.keras.preprocessing.image_dataset_from_directory(
                    "tmp",
                    shuffle=True,
                    seed=RANDOM_STATE,
                    label_mode="categorical",
                    image_size=(640, 360),
                )

                # to Numpy
                print("TF Data to Numpy")
                for kkk in game_play.as_numpy_iterator():
                    tmp = kkk
                x, y = kkk
                del tmp, kkk, game_play

                # x = np.concatenate([x, ], axis=1)
                print(x.shape, y.shape)
                Q_net.QNet(
                    x,
                    tf.keras.activations.tanh(
                        tf.nn.softmax([float(play_time), 85.0])),
                    y,
                )

                # dqn.fit(x, callbacks=[tf.keras.callbacks.TensorBoard(log_dir="logs/fit/play/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)])
                # dqn.predict(x)
                # dqn.save('Model\\Play\\game_play.h5')
                # except:
                #     pass

                isStart = 0
                # shutil.rmtree("tmp")
                # time.sleep(0)

                Retry()
            else:
                print(
                    "This may issue is an issue where AI is slow to detect the image on the screen."
                )
            # time.sleep(0.42)


def GamePlay():
    np.set_printoptions(suppress=True)

    # model = load_model("../Model/Keras/keras_model.h5", custom_objects=None)

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
        PlayWithLearning()

    elif First_State == 4:
        GamePlay()

    elif First_State == 3:
        train_dataset, validation_dataset = ImportImageDataSet()

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
