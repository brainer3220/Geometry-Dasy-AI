import pyautogui as pag
import mss, cv2
import tensorflow as tf
import numpy as np

from PIL import image   #이미지 처리 라이브러리
from tensorflow.keras import layers

Play_Button = [361, 228]    #첫 시작 버튼
next_Button = [980, 245]    # 게임내 목록 다음 버튼
Previous = [42, 245]        #게임내 모곡 이전 버튼

result = 0  #결과 반환 값, 추후 무슨 일을 할것인지 반환

while True:
    x, y = pag.position()
    position_str = 'X: ' + str(x) + 'Y: '+ str(y)
    print(position_str)

    def compute_icon_type(img):
        mean = nm.mean(img, axis=(0, 1))

        if mean[0] > 50 and mean[0] < 55 and mean[1] > 50 and mean[1] < 55 and mean[2] > 50 and mean[2] < 55:
            result = "Play_Button"

        else if mean[0] > 250 and mean[1] > 85 and mean[1] < 110 and mean[2] > 250:
            result = "Lock"
            
with mss.mss() as sct:


model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layersDense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 with 10 output units:
model.add(layers.Dense(10, activation='softmax'))


Play_Button = compute_icon_type(Play_button)

display[10][10] #기기 화면의 캐릭터 주변을 인식할 행렬
reward  #강화 학습을 위한, 보상

if result == 0:

elif result == 1:

elif result == 2:
