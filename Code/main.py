import tensorflow as tf
import pyautogui as pag
import mss, cv2
import numpy as np

from PIL import Image


    def compute_icon_type(img):
        mean = nm.mean(img, axis=(0, 1))
        if mean[0] > 50 and mean[0] < 55 and mean[1] > 50 and mean[1] < 55 and mean[2] > 50 and mean[2] < 55:

        else if mean[0] > 250 and mean[1] > 85 and mean[1] < 110 and mean[2] > 250:
            result = "Lock"
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layersDense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 with 10 output units:
model.add(layers.Dense(10, activation='softmax'))
# Mouse Click Function
# Funciton

# Jump Function
def jump():
        pag.press("space")

# Click to Start Button
def Click_Start():
        pag.moveTo(417, 257)    # X and y coordinates of the start button
        pag.mouseDown()
        pag.mouseUp()

Play_Button = compute_icon_type(Play_button)

display[10][10] #기기 화면의 캐릭터 주변을 인식할 행렬
reward  #강화 학습을 위한, 보상

if result == 0:

elif result == 1:
# Function to get resolution.
# Test it when you bring the emulator's resolution coordinates.
while True:
   x, y = pag.position()
   position_str = 'X: ' + str(x) + 'Y: ' + str(y)
   print(position_str)

elif result == 2:
