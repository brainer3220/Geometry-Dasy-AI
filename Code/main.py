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

                cv2.imshow('Game_Src', Game_Scr)
                cv2.waitKey(0)

