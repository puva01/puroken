# -*- coding: utf-8 -*-
#include "windows.h"
import cv2
import numpy as np
import time
from operator import itemgetter
import wiringpi
import sys


#射影変換行列を計算するキャリブレーション用関数
def calibration(areas):
    pts1 = np.float32([[446,171],[790,173],[446,514],[788,514]]) #areasと同じ順　(左上，右上，左下，右下)
    #areasの格納が普通の配列じゃないので，配列に書き直す．
    x0=areas[0][0][0][0]
    y0=areas[0][0][0][1]
    x1=areas[0][1][0][0]
    y1=areas[0][1][0][1]
    x2=areas[0][2][0][0]
    y2=areas[0][2][0][1]
    x3=areas[0][3][0][0]
    y3=areas[0][3][0][1]
    p0 = [x0,y0]
    p1 = [x1,y1]
    p2 = [x2,y2]
    p3 = [x3,y3]
    area = [p0,p1,p2,p3]
    #areasの4つの座標を左上，右上，左下，右下の順で並び替える．台形の場合x座標の順で一意に決定
    #np.float32リストに対してitemgetterは使えないらしい
    area.sort(key=itemgetter(0))
    #スケール変換変換
    k = (pts1[3][0]-pts1[2][0])/(area[2][0]-area[1][0])
    h = (area[1][1]-area[0][1])*k
    delta = (area[1][0]-area[0][0])*k
    x1 = pts1[2]+[-delta,-h1]
    x2 = pts1[3]+[delta,-h2]
    pts2 = np.float32([x1,x2,pts1[2],pts1[3]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return M

#台形補正画像を生成する関数
def revision(M,img):
    inv_M =np.linalg.inv(M)
    dst = cv2.warpPerspective(img,inv_M,(0,800))
    return dst

def getBlue(frame):
    # Convert BGR to HSV and smooth
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    smooth=cv2.GaussianBlur(hsv,(15,15),0)

    # define range of blue color in HSV　(第1引数を110〜130→90〜140に変更)
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([140,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(smooth, lower_blue, upper_blue)

    # Bitwise-AND mask and original image(白黒画像の中で，白の部分だけ筒抜けになって映る)
    res = cv2.bitwise_and(image,image, mask= mask)
    image,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    contours.sort(key=cv2.contourArea,reverse=True)
    if  contours:
        cnt = contours[0]
        epsilon = 0.08*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        areas.append(approx)
    else:
        cnt = []
    return areas,res,cnt,



calibration_img = cv2.imread("calibration.png",1)

button_pin1 = 17 # 11番端子
button_pin2 = 27 # 11番端子
button_pin3 = 22 # 11番端子
# GPIO初期化
wiringpi.wiringPiSetupGpio()
# GPIOを出力モード（1）に設定
wiringpi.pinMode( button_pin1, 0 )
wiringpi.pinMode( button_pin2, 0 )
#wiringpi.pinMode( button_pin3, 0 )
# 端子に何も接続されていない場合の状態を設定
# 3.3Vの場合には「2」（プルアップ）
wiringpi.pullUpDnControl( button_pin1, 2 )
wiringpi.pullUpDnControl( button_pin2, 2 )
#wiringpi.pullUpDnControl( button_pin3, 2 )




capture = cv2.VideoCapture(0)

count = 0
count2 = 0
l = []
m = []
M = []
# isOpenedの代わりにTrueを使うと，frameがemptyのときエラーを吐く
while capture.isOpened():
    ret, frame = capture.read()
    if ret :
        cv2.imshow('frame',frame)
        #キャリブレーションボタンが押された場合(右)，射影変換行列を計算
        if wiringpi.digitalRead(button_pin1) == 0:
            While count2<6:
                cv2.imshow('img',calibration_img)
                areas,_,_=getBlue(frame)
                M = calibration(areas)
                count2 = count2+2
        #スイッチ1が押された場合．
        elif wiringpi.digitalRead(button_pin2) == 0:
            #キャリブレーションボタンを押していた場合，画像を台形補正．
            if　M:
                img1 = revision(M,img1)
            #frameから輪郭をとる
            cv2.imshow('img',back1)

    if cv2.waitKey(-1) &  0xFF == ord('q'):
        break
