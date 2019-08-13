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




# Function to get resolution.
# Test it when you bring the emulator's resolution coordinates.
while True:
   x, y = pag.position()
   position_str = 'X: ' + str(x) + 'Y: ' + str(y)
   print(position_str)

