#coding: UTF-8
#include "windows.h"
import cv2
import numpy as np
import time
from operator import itemgetter

#オリジナル画像に対応した台形座標取得
def getPts(areas):
    pts1 = np.float32([[120,36],[280,36],[120,190],[280,190]])
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
    k = (pts1[3][0]-pts1[2][0])/(area[3][0]-area[1][0])
    h1 = (area[0][1]-area[1][1])*k
    h2 = (area[3][1]-area[2][1])*k
    delta1 = (area[0][0]-area[1][0])*k
    delta2 = (area[2][0]-area[3][0])*k
    x1 = pts1[2]+[-delta1,-h1]
    x2 = pts1[3]+[delta2,-h2]
    pts2 = np.float32([x1,x2,pts1[2],pts1[3]])
    return pts1,pts2,x1,x2,k,delta1,h1,h2,area

#台形補正画像を生成する関数
def revision(pts1,pts2,img):
    M = cv2.getPerspectiveTransform(pts1,pts2)
    inv_M =np.linalg.inv(M)
    dst = cv2.warpPerspective(img,inv_M,(600,400))
    return dst

#青の輪郭を取ってくる関数
def getBlue(image):
    # Convert BGR to HSV and smooth
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    smooth=cv2.GaussianBlur(hsv,(15,15),0)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(smooth, lower_blue, upper_blue)

    # Bitwise-AND mask and original image(白黒画像の中で，白の部分だけ筒抜けになって映る)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    image,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    contours.sort(key=cv2.contourArea,reverse=True)
    cnt = contours[0]
    epsilon = 0.08*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    # areas.append(np.array(approx))
    areas.append(approx)
    return (areas,res)

img = cv2.imread("blue.png",1)
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)


capture = cv2.VideoCapture(0)

count = 0
# isOpenedの代わりにTrueを使うと，frameがemptyのときエラーを吐く
while capture.isOpened():
    ret, frame = capture.read()

    if ret :
        areas,res = getBlue(frame)
        cv2.drawContours(res, areas, -1, (0,0,255), 3)
        cv2.imshow('frame',frame)
        cv2.imshow('res',res)
        if len(areas[0])==4 :
            pts1,pts2,x1,x2,k,delta1,h1,h2,area = getPts(areas)
            print ('h1:{}'.format(h1))
            print ('h2:{}'.format(h2))
            print ('area:{}'.format(area))
            print ('pts2:{}'.format(pts2))
            dst = revision(pts1,pts2,img)
            cv2.imshow('img',dst)

#waitKeyの引数を0以下にするとキー入力する毎に画面がframeが更新する．
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# cv2.imshow('img',img)
# cv2.waitKey(-1)
