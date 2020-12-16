import glob
import os
import time
import datetime

import cv2
import mss
import pyautogui as pag
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.models import load_model

from PIL import Image, ImageOps
from PIL import Image
from PIL import ImageGrab


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
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
    if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
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
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
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
        position_str = 'X: ' + str(x) + 'Y: ' + str(y)
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

            Game_Scr = cv2.resize(Game_Scr,
                                  dsize=(960, 540),
                                  interpolation=cv2.INTER_AREA)
            # Game_Scr = np.ravel(Game_Scr)

            GMD_Model = os.path.join(os.getcwd(), "Model", "CNN",
                                     "saved_model.pb")
            GMD_Model_Keras = os.path.join(os.getcwd(), "..", "Model", "Keras",
                                           "keras_model.h5")

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
                            "..\Graph\GMDmiss", graph=tf.get_default_graph())
                        writer.close()
            print(Gmd.shape)
            print(Gmd)
            # cv2.imshow('Game_Src', Game_Scr)
            # cv2.waitKey(0)

            # CNN
            # model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
            # model.add(Conv2D(64, (3, 3), activation='relu'))



def Vidio_Analyze(Video):
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


First_State = int(
def BinaryImageClassf():
    model = Sequential()
    model.add(Conv2D(120, 60, 3, padding='same', activation='relu',
                        input_shape=(640, 360, 3)))
    model.add(MaxPooling2D(pool_size=(65, 25)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(60, 30, 3, padding='same'))
    model.add(MaxPooling2D(pool_size=(60, 25), padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(60, 25, 3, padding='same'))
    model.add(MaxPooling2D(pool_size=(60, 25), padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'sigmoid'))

    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model
    input("""If you want to analyze your video?
press 1.

or real time play game and real time screen analyze.
press 2.

If learning Geometry Dash Miss image
Press 3.

If you gaming from real time
Press 4
"""))

if First_State == 1:
    Video = input("Please enter a video path and video name.")
    Vidio_Analyze(Video)
elif First_State == 2:
    Real_Time()
elif First_State == 4:
    Game_play()

elif First_State == 3:
    GmdMiss_Folder = os.path.join(os.getcwd(), "..", "Photo", "GMD Miss")
    GMD_Play_Folder = os.path.join(os.getcwd(), "..", "Photo", "GMD_Play")
    GmdMiss_List = os.listdir(GmdMiss_Folder)
    GMD_Play_List = os.listdir(GMD_Play_Folder)

    # Test that the file is read correctly
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    Batch_Size = 30
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, '..\model\CheckPoint\GMDmissData')
    GMD_Miss_Y = [0, 0, 1]
    GMD_Miss_Y = np.tile(GMD_Miss_Y, (len(GmdMiss_List), 1))
    print(GMD_Miss_Y)
    Img_Miss_List = []
    Img_Play_List = []

    # print(np.array(cv2.imread(os.path.join(os.getcwd(), GmdMiss_Folder, GmdMiss_List[1]), cv2.IMREAD_GRAYSCALE)))

    for i in range(0, len(GmdMiss_List)):
        print(i)
        Img = os.path.join(os.getcwd(), GmdMiss_Folder, GmdMiss_List[i])
        Img = cv2.imread(Img)
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = np.array(Img)
        Img = cv2.resize(Img, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
        Img_Miss_List.append(Img)

    # for i in range(0, len(GMD_Play_List)):
    #     print(i)
    #     Img = os.path.join(os.getcwd(), GMD_Play_Folder, GMD_Play_List[i])
    #     Img = cv2.imread(Img)
    #     Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    #     Img = np.array(Img)
    #     Img = cv2.resize(Img, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
    #     Img_Play_List.append(Img)
    i = 0
    bias = np.ones((1, 1), dtype=float)
    while True:
        print(i)
        Img = Img_Miss_List[i]
        print(Img)
        # Img = tf.reshape(Img, [4, 1])
        print(Img)
        # Img = cv2.resize(Img, dsize=(960, 540), interpolation=cv2.INTER_AREA)
        with tf.Session() as sess:
            graph = tf.Graph()
            with graph.as_default():
                with tf.name_scope("Convolution"):
                    Img = Convolution(Img)
                with tf.name_scope("Relu_Function"):
                    Img = tf.nn.relu(Img)
                with tf.name_scope("MaxPool"):
                    Img = Max_Pool(Img)
                    print(Img.shape)
                with tf.name_scope("Img_Fatten"):
                    Img_Flatten = tf.reshape(Img, [-1, 30 * 58 * 3])
                with tf.name_scope("Fully_Connected"):
                    X = Img_Flatten  # img is X
                with tf.name_scope("Output_layer"):
                    # X = tf.placeholder(tf.float32, shape=[None, 30*58*3])
                    Y = tf.placeholder(tf.float32, shape=[None, 3])
                    W = tf.Variable(tf.zeros(shape=[30 * 58 * 3, 3]))
                    B = tf.Variable(tf.zeros(shape=[3]))

                    with tf.name_scope("Logits"):
                        Logits = tf.matmul(Img_Flatten, W) + B
                    with tf.name_scope("SoftMax"):
                        Y_Pred = tf.nn.softmax(Logits)

                #     lables is state num.
                #     0: Nothing
                #     1: Game play screen
                #     2: Game over screen

                with tf.name_scope("Learning"):
                    with tf.name_scope("Reduce_Mean"):
                        Loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=Logits, labels=GMD_Miss_Y))
                    # with tf.name_scope("TrainStep"):
                    #     Train_Step = tf.train.GradientDescentOptimizer(0.5).minimize(Loss)
                    with tf.name_scope("Optimizer"):
                        Optimizer = tf.train.AdamOptimizer(Learning_Rate)
                    with tf.name_scope("Train"):
                        Train = Optimizer.minimize(loss=Loss)
                    with tf.name_scope("Argmax_Compare"):
                        Predictive_Val = tf.equal(tf.argmax(Y_Pred, 1),
                                                  tf.argmax(GMD_Miss_Y, 1))
                    with tf.name_scope("Accuracy"):
                        Accuracy = tf.reduce_mean(
                            tf.cast(Predictive_Val, dtype=tf.float32))

                i += 1

                if i == len(GmdMiss_List):
                    writer = tf.summary.FileWriter(
                        "..\Graph\GMDmiss", graph=tf.get_default_graph())
                    print(Img)
                    print(i)
                    saver.save(
                        save_path="F:\Programing\Geomatry-Dasy-AI\Model\CNN",
                        global_step=i,
                    )
                    writer.close()

                    for k in range(1000):
                        sess.run(fetches,
                                 feed_dict=None,
                                 options=None,
                                 run_metadata=None)
                    break

            start_time = datetime.now()
            # for k in range(30):
            #     Total_Batch = int(len(GmdMiss_List) / Batch_Size)
            # for Step in range(Total_Batch):
            #     Loss_Val, _ = sess.run([Loss, Train], feed_dict={X: Img_Miss_List, Y: GMD_Miss_Y})
            # if k % 100 == 0:
            #     print("Epoch = ", i, ",Step =", Step, ", Loss_Val = ", Loss_Val)
            # End_Time = datetime.now()
            #     saver.save(sess=sess, save_path='..\Model\GMDmissLearningData', global_step=None)
            print(i)
            if i == len(Img_Miss_List):
                break
                # # Accuracy 확인
                # test_x_data = mnist.test.images    # 10000 X 784
                # test_t_data = mnist.test.labels    # 10000 X 10
                # accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, T: test_t_data})
                # print("\nAccuracy = ", accuracy_val)
