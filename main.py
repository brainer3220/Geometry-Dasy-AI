import pyautogui as pag
import mss, cv2
import tensorflow as tf
import numpy as np

def compute_icon_type(img):
    mean = nm.mean(img, axis=(0, 1))

    if mean[0] > 50 and mean[0] < 55 and mean[1] > 50 and mean[1] < 55 and mean[2] > 50 and mean[2] < 55:
        result = "Play_Button"

    else if mean[0] > 250 and mean[1] > 85 and mean[1] < 110 and mean[2] > 250:
        result = "Lock"

Play_Button = compute_icon_type(Play_button)

display[10][10] #기기 화면의 캐릭터 주변을 인식할 행렬
reward  #강화 학습을 위한, 보상

if
