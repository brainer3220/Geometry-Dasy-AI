import tensorflow as tf
import pyautogui as pag
import mss
import cv2
import numpy as np
import time
import os
import glob
from datetime import datetime      # datetime.now() 를 이용하여 학습 경과 시간 측정


from PIL import Image
from PIL import ImageGrab

saver = tf.train.Saver

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
    # return

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
    GmdMiss_Folder = os.path.join(os.getcwd(), '..', 'Photo', 'GMD Miss')
    GMD_Play_Folder = os.path.join(os.getcwd(), '..', 'Photo', 'GMD_Play')
    GMD_Play_Folder = os.listdir(GMD_Play_Folder)
    GmdMiss_List = os.listdir(GmdMiss_Folder)
    print(GmdMiss_List)
    Img = os.path.join(os.getcwd(), GmdMiss_Folder, GmdMiss_List[0])
    print(GmdMiss_List[0])
    Img = cv2.imread(Img, cv2.IMREAD_GRAYSCALE)
    Img = np.array(Img)
    print(Img)
    print(len(GmdMiss_List))

    # Test that the file is read correctly
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    Batch_Size = 30
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, '..\model\CheckPoint\GMDmissData')
    GMD_Miss_Y = [0,0,1]
    GMD_Miss_Y = np.tile(GMD_Miss_Y, (3,1))
    print(GMD_Miss_Y)
    Img_Miss_List = []

    # print(np.array(cv2.imread(os.path.join(os.getcwd(), GmdMiss_Folder, GmdMiss_List[1]), cv2.IMREAD_GRAYSCALE)))

    for i in range(0, len(GmdMiss_List)):
        print(i)
        Img = os.path.join(os.getcwd(), GmdMiss_Folder, GmdMiss_List[i])
        print(GmdMiss_Folder)
        print(Img)
        Img = cv2.imread(Img, cv2.IMREAD_GRAYSCALE)
        Img = np.array(Img)
        print(Img.shape)
        Img_Miss_List.append(Img)
    i = 0
    bias = np.ones((1, 1), dtype=float)
    while True:
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
                    Img_Flatten = tf.reshape(Img, [-1, 30, 58, 1])
                # with tf.name_scope("Fully_Connected"):
                #     X = tf.reshape(Img, [-1, 30*58*1])    # img is X
                with tf.name_scope("Output_layer"):
                    X = tf.placeholder(tf.float32, shape=[None, 1740])
                    Y = tf.placeholder(tf.float32, shape=[None, 3])
                    W = tf.Variable(tf.zeros(shape=[30*58*1, 3]))
                    B = tf.Variable(tf.zeros(shape=[3]))

                with tf.name_scope("Logits"):
                    Logits = tf.matmul(X, W) + B

                with tf.name_scope("SoftMax"):
                    Y_Pred = tf.nn.softmax(Logits)

            #     lables is state num.
            #     0: Nothing
            #     1: Game play screen
            #     2: Game over screen

                with tf.name_scope("Learning"):
                    with tf.name_scope("Reduce_Mean"):
                        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Logits, labels=GMD_Miss_Y))
                    # with tf.name_scope("TrainStep"):
                    #     Train_Step = tf.train.GradientDescentOptimizer(0.5).minimize(Loss)
                    with tf.name_scope("Optimizer"):
                        Optimizer = tf.train.AdamOptimizer(Learning_Rate)
                    with tf.name_scope("Train"):
                        Train = Optimizer.minimize(loss=Loss)
                    with tf.name_scope("Argmax_Compare"):
                        Predictive_Val = tf.equal(tf.argmax(Y_Pred, 1), tf.argmax(GMD_Miss_Y, 1))
                    with tf.name_scope("Accuracy"):
                        Accuracy = tf.reduce_mean(tf.cast(Predictive_Val, dtype=tf.float32))

                if i == 1:
                    writer = tf.summary.FileWriter('..\Graph\GMDmiss', graph=tf.get_default_graph())
                    print(Img)
                    print(i)
                    writer.close()

            start_time = datetime.now()
            for k in range(30):
                Total_Batch = int(len(GmdMiss_List) / Batch_Size)
                for Step in range(Total_Batch):
                    Loss_Val, _ = sess.run([Loss, Train], feed_dict={X: Img_Miss_List, Y: GMD_Miss_Y})
                    if Step % 100 == 0:
                        print("Epoch = ", i, ",Step =", Step, ", Loss_Val = ", Loss_Val)
                End_Time = datetime.now()
                    # saver.save(sess=sess, save_path='..\Model\GMDmissLearningData', global_step=None)
            i += 1
            print(i)
            if i == len(Img_Miss_List):
                break
                    # # Accuracy 확인
                    # test_x_data = mnist.test.images    # 10000 X 784
                    # test_t_data = mnist.test.labels    # 10000 X 10
                    # accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, T: test_t_data})
                    # print("\nAccuracy = ", accuracy_val)
