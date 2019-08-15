import tensorflow as tf
import pyautogui as pag
import mss, cv2
import numpy as np

from PIL import Image

# Funciton

# Jump Function
def jump():
        pag.press("space")

# Click to Start Button
def Click_Start():
        pag.moveTo(417, 257)    # X and y coordinates of the start button
        pag.mouseDown()
        pag.mouseUp()

def bring_window():
        time.sleep(0.5)
        apple = """
        tell application "BlueStacks"
        activate
        end tell
        """
def Convolution(img):
        kernel = tf.Variable(tf.truncated_normal(shape=[250, 250, 3, 3], stddev=0.1))
        sess = tf.Session()
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                img = img.astype('float32')
                conv2d = tf.nn.conv2d(np.expand_dims(img, 0), kernel, strides=[1, 25, 25, 1], padding='VALID')  # + Bias1
                conv2d = sess.run(conv2d)
                return conv2d



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
while True:
        with mss.mss() as sct:
                Game_Scr = np.array(sct.grab(Game_Scr_pos))[:,:,:3]

                # Below is a test to see if you are capturing the screen of the emulator.
                # cv2.imshow('Game_Src', Game_Scr)
                # cv2.waitKey(0)

                Game_Scr = cv2.resize(Game_Scr, dsize=(960, 540), interpolation=cv2.INTER_AREA)
                print(Convolution(Game_Scr))    # CNN Results
