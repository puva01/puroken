# -*- coding: utf-8 -*-
#include "windows.h"
import cv2
import time

img1 = cv2.imread("right.jpg")
img2 = cv2.imread("left.jpg")
img3 = cv2.imread("white.jpg")
#フルスクリーン表示
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)


# GPIO出力モードを1に設定する
wiringpi.wiringPiSetupGpio()
# GPIOを入力モード(0)にする
wiringpi.pinMode( switch_pin, 0 )
wiringpi.pinMode( switch_pin2, 0 )

while True:
    # GPIO端子の状態を読み込み
    # 0V : 0
    # 3.3V : 1
    if wiringpi.digitalRead(switch_pin) == 1 :
        print ("Switch ON")
        cv2.imshow("img",img1)
        cv2.waitKey(500)
    elif( wiringpi.digitalRead(switch_pin2) == 1)
        print ("Switch OFF")
        cv2.imshow("img",img2)
        cv2.waitKey(500)
    # 1秒ごとに検出
    time.sleep(10)

