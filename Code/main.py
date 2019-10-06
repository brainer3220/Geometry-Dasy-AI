import tensorflow as tf
import pyautogui as pag
import mss
import cv2
import numpy as np
import time
import os
import glob

from PIL import Image
from PIL import ImageGrab
from keras.models import Sequential

saver = tf.train.Saver()

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

# Funciton

# Jump Function


def Jump():
        pag.press("space")

# Q Value Function


def Q_Value(State, Action):
        reword + (Discount * max(Q_next))
        return

# Click to Start Button


def Click_Start():
        pag.moveTo(417, 257)    # X and y coordinates of the start button
        pag.mouseDown()
        pag.mouseUp()

# Bring the emulator to the front


def bring_window():
        time.sleep(0.5)
        apple = """
        tell application "BlueStacks"
        activate
        end tell
        """


def average_hash(fname, size=16):
        img = Image.open(fname)
        img = img.convert('L')
        img = img.resize((960, 540), Image.ANTIALIAS)
        pixel_data = img.getdata()
        pixels = np.array(pixel_data)
        pixels = pixels.reshape((960, 540))
        avg = pixels.mean()
        diff = 1 * (pixels > avg)
        print(diff)


def Convolution(img):
        kernel = tf.Variable(tf.truncated_normal(shape=[180, 180, 3, 3], stddev=0.1))
        # Gray_Scale(img)
        img = img.astype('float32')
        # print(img.shape)
        img = tf.nn.conv2d(np.expand_dims(img, 0), kernel, strides=[ 1, 15, 15, 1], padding='VALID')  # + Bias1
        return img

def Max_Pool(img):
        img = tf.nn.max_pool(img, ksize=[1,2,2,1] , strides=[1,2,2,1], padding='VALID')
        return img


Pixel_X = tf.placeholder(tf.float32, [None, 128, 128])

# Function to get resolution.
# Test it when you bring the emulator's resolution coordinates.
# while True:
#    x, y = pag.position()
#    position_str = 'X: ' + str(x) + 'Y: ' + str(y)
#    bring_window()
#    print(position_str)

# Full resolution of the emulator
Game_Scr_pos = {'left': 16, 'top': 54, 'height': 483, 'width': 789}

# Where to click the button on the emulator.
Game_Src_Click_pos = [379, 283]

sess = tf.Session()

# Gray Scale


def Gray_Scale(img):
        tf.image.rgb_to_grayscale(
            img,
            name=None)

def Real_Time():
        bring_window()
        i = 0
        while True:
                i+=1
                with mss.mss() as sct:
                    Game_Scr = np.array(sct.grab(Game_Scr_pos))[:, :, :3]

                    # Below is a test to see if you are capturing the screen of the emulator.
                    # cv2.imshow('Game_Src', Game_Scr)
                    # cv2.waitKey(0)

                    Game_Scr = cv2.resize(Game_Scr, dsize=(960, 540), interpolation=cv2.INTER_AREA)
                    # Game_Scr = np.ravel(Game_Scr)
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
                                writer = tf.summary.FileWriter('..\Graph\GMDmiss', graph=tf.get_default_graph())
                                writer.close()
                    print(Gmd.shape)
                    print(Gmd)
                    # cv2.imshow('Game_Src', Game_Scr)
                    # cv2.waitKey(0)

                    # CNN
                    # model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
                    # model.add(Conv2D(64, (3, 3), activation='relu'))


# loss = tf.reduce_mean(tf.square(y-Q_action))
# Optimizer = tf.trainAdamsOptimizer(learning_rate)
# training_op = optimizer.minize(loss)

def Vidio_Analyze(Video):
        Vidcap = cv2.VideoCapture(Video)
        success, image = Vidcap.read()
        count = 0
        while success:
            # save frame as JPEG file
            cv2.imwrite("frame%d.jpg" % count, image)
            success, image = Vidcap.read()
            print('Read a new frame: ', success)
            count += 1

def Game_Play_With_Learning():
        Num_Of_Play_Time = int(input("Press number from Game Time."))
        while True:
                Real_Time()
                Play_Time = time.time() # Game start time
                Jump()

                # if:        # if ended from One of game, up to Play Time.
                #
                #     Num_Of_Play_Time += 1
                #     Play_Time = time.time() - Play_Time # Playtime for one game

                if epoch > Play_Time:
                        break
# def Play_Learning:


# def Game_Replay():
#     while True:


First_State = int(input("""If you want to analyze your video?
press 1.

or real time play game and real time screen analyze.
press 2.

If learning Geometry Dash Miss image
Press 3.
"""))

if First_State == 1:
        Video = input("Please enter a video path and video name.")
        Vidio_Analyze(Video)
elif First_State == 2:
        Real_Time()
elif First_State == 3:
        Img_Folder = os.path.join(os.getcwd(), '..', 'Photo', 'GMD Miss')
        File_List = os.listdir(Img_Folder)
        print(File_List)
        img = os.path.join(os.getcwd(), Img_Folder, File_List[0])
        print(File_List[0])
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        print(img)
        print(len(File_List))

        # Test that the file is read correctly
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        Batch_Size = 30
        i = 0
        GMD_Miss_Learning = tf.global_variables_initializer
        while True:
            if i in range(len(Img_Folder)):
                img = File_List[i]
                img = os.path.join(os.getcwd(), Img_Folder, File_List[i])
                img = cv2.imread(img)
                with tf.Session() as sess:
                    graph = tf.Graph()
                    with graph.as_default():
                        with tf.name_scope("Convolution"):
                            img = Convolution(img)
                        with tf.name_scope("Relu_Function"):
                            img = tf.nn.relu(img)
                        with tf.name_scope("MaxPool"):
                            img = Max_Pool(img)
                            print(img.shape)
                        with tf.name_scope("Fully_Connected"):
                            img = tf.reshape(img, [-1, 30*58*3])
                        with tf.name_scope("Output_layer"):
                            W = tf.Variable(tf.random_normal([30*58*3, 3], stddev=0.01))
                            B = tf.Variable(tf.random_normal([3]))
                            with tf.name_scope("Linear_Regression"):
                                img = tf.matmul(img, W)
                            with tf.name_scope("SoftMax"):
                                img = tf.nn.softmax(img)

                        if i%20 == 0:
                            saver.save(sess, '..\Learning\CheckPoint', Learning_Step = i)
                        if i == 1:
                            writer = tf.summary.FileWriter('..\Graph\GMDmiss', graph=tf.get_default_graph())
                            print(img)
                            print(i)
                            writer.close()
                i += 1
            else:
                break
