import tensorflow as tf
import pyautogui as pag
import mss
import cv2
import numpy as np
import time
import os, glob

from matplotlib import pyplot as plt # as는 alias 적용시 사용

from PIL import Image
from PIL import ImageGrab
from keras.models import Sequential

epsilon = 1  # Random probability
epsilon_Minimum_Value = 0.001  # epsilon의 최소값
nbActions = 2  # Number of actions (jump, wait)
epoch = 1001  # Game repeat count
hidden_Size = 100  # Hidden layer count
max_Memory = 5000  # Maximum number of game contents remembered
batch_Size = 50  # Number of data bundles in training
grid_Size = 10  # Grid size
nb_States = grid_Size * grid_Size  # State count
Discount = 0.9  # discount Value
learning_Rate = 0.2  # Learning_Rate

Replay_Meomry = 100000

reword = 0

# Funciton

# Jump Function


def jump():
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
        kernel = tf.Variable(tf.truncated_normal(
            shape=[250, 250, 3, 3], stddev=0.1))
        with tf.Session() as sess:
                Gray_Scale(img)
                sess.run(tf.global_variables_initializer())
                img = img.astype('float32')
                img = tf.nn.conv2d(np.expand_dims(img, 0), kernel, strides=[
                                   1, 30, 30, 1], padding='VALID')  # + Bias1
                img = sess.run(img)
                img = tf.nn.relu(img)
                img = sess.run(img)
                Max_Pool(img)
                return img
# Max Pooling


def Max_Pool(img):
        with tf. Session() as sess:
                img = tf.nn.max_pool(img, ksize=[1, 250, 250, 3], strides=[
                                     1, 25, 25, 3], padding='SAME')
                sess.run(tf.global_variables_initializer())
                img = sess.run(img)
                img = img.eval()
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
        while True:
                with mss.mss() as sct:
                        Game_Scr = np.array(sct.grab(Game_Scr_pos))[:, :, :3]

                        # Below is a test to see if you are capturing the screen of the emulator.
                        cv2.imshow('Game_Src', Game_Scr)
                        cv2.waitKey(0)

                        Game_Scr = cv2.resize(Game_Scr, dsize=(
                            960, 540), interpolation=cv2.INTER_AREA)
                        # Game_Scr = Game_Scr.resize((960, 540))
                        # Game_Scr = np.ravel(Game_Scr)

                        # print(Gmd.Convolution(Game_Scr))    # CNN Results
                        Gmd = Convolution(Game_Scr)
                        print(Gmd)

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


First_State = int(input("""If you want to analyze your video?
press 1.

or real time play game and real time screen analyze.
press 2.

GMD Miss Learning
press 3.
"""))

if First_State == 1:
        Video = input("Please enter a video path and video name.")
        Vidio_Analyze(Video)
elif First_State == 2:
        Real_Time()
elif First_State == 3:
        i = 0
        while True:
                Img_Folder = os.path.join(os.getcwd(), '..', 'Photo', 'GMD Miss');
                File_List = os.listdir(Img_Folder)
                print(File_List)
                img = Img_Folder + File_List[0]
                print(img)
                img = cv2.imread(img)
                # img = Img_Folder + File_List[0]
                print(img)
                img = np.array(img)
                print(img)
                cv2.imshow('img', img)
                cv2.waitkey(0)
                cv2.destroyAllWindows()
                if len(Img_Folder) > i:
                        img = Img_Folder + File_List[i]
                        img = cv2.imread(File_List[i], cv2.IMREAD_GRAYSCALE)
                        kernel = tf.Variable(tf.truncated_normal(shape=[250, 250, 3, 3], stddev=0.1))
                        kernel = tf.Variable(tf.random.truncated_normal(shape=[250, 250, 3, 3], stddev=0.1))
                        with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                # img = img.astype('float32')
                                img = tf.nn.conv2d(np.expand_dims(img, 0), kernel, strides=[1, 30, 30, 1], padding='VALID')  # + Bias1
                                img = sess.run(img)
                                img = tf.nn.relu(img)
                                # img = sess.run(img)
                                activation_map = sess.run(
                                        tf.minimum(tf.nn.relu(img), 255))
                                Max_Pool(img)
                                sess = tf.Session()
                                saver.save(zsess, '..model\CNN\GMD_Miss\GMDmiss')
                                i += 1
                else:
                        break
