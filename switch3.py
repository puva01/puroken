# -*- coding: utf-8 -*-
#include "windows.h"

# GPIOを制御するライブラリ
import wiringpi
# タイマーのライブラリ
import time
import cv2
import sys
# ボタンを繋いだGPIOの端子番号
button_pin1 = 17 # 11番端子
button_pin2 = 27 # 11番端子

img1 = cv2.imread("right2.jpg")
img2 = cv2.imread("turnright2.jpg")
img3 = cv2.imread("switchoff-1.jpg")
# GPIO初期化
wiringpi.wiringPiSetupGpio()
# GPIOを出力モード（1）に設定
wiringpi.pinMode( button_pin, 0 )
# 端子に何も接続されていない場合の状態を設定
# 3.3Vの場合には「2」（プルアップ）
# 0Vの場合は「1」と設定する（プルダウン）
wiringpi.pullUpDnControl( button_pin, 2 )
#cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
# cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

# whileの処理は字下げをするとループの範囲になる（らしい）
while True:
    # GPIO端子の状態を読み込む
    # ボタンを押すと「0」、放すと「1」になる
    # GPIOの状態が0V(0)であるか比較
    if( wiringpi.digitalRead(button_pin1) == 0 ):
        # 0V(0)の場合に表示
        print ("Switch1 ON")
        cv2.imshow('img',img1)
        cv2.waitKey(1000)
    if( wiringpi.digitalRead(button_pin2) == 0 ):
        print ("Swich2 ON")
        cv2.imshow('img',img2)
        cv2.waitKey(1000)
    else:
        print ("Switch OFF")
        cv2.imshow('img',img3)
        cv2.waitKey(1000)
    time.sleep(0.05)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break
