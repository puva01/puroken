#coding: UTF-8
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

#ターゲット(黄色)の輪郭を取ってくる関数
def getTarget(image):
    # Convert BGR to HSV and smooth
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    smooth=cv2.GaussianBlur(hsv,(15,15),0)
    #黄色
    lower_blue = np.array([25,100,85])
    upper_blue = np.array([100,255,255])

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

#台形補正用にwebcameraから輪郭をとってくる．
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

#輪郭の重心を計算
def center_of_image(image):
    areas,_,_ = getTarget(image)
    if areas:
        M = cv2.moments(areas[0])
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
    else:
        x=0
        y=0
    return x,y


#backをforeを中心から(x,y)移動させて重ね合わせる
def clip_image(x, y, back, fore):
    h1, w1, _ = back.shape
    h2, w2, _ = fore.shape
    X = int((w1-w2)/2)
    Y = int((h1-h2)/2)
    if abs(x)< X and abs(y)<Y:
        back[Y+y:Y+h2+y,X+x:X+w2+x] = fore
        #+xではみ出す，
    elif x >=X and abs(y)<Y:
        back[Y+y:Y+h2+y,X+X:X+w2+X,] = fore
        #-xではみ出す，
    elif -x >=X and abs(y)<Y:
        back[Y+y:Y+h2+y,0:w2] = fore
        #+yではみ出す，
    elif abs(x)<X and y>=Y:
        back[Y+Y:Y+h2+Y,X+x:X+w2+x] = fore
        #-yではみ出す，
    elif abs(x)<X and -y>Y:
        back[0:h2,X+x:X+w2+x] = fore
        #+x，+yではみ出す，
    elif x >=X and y>=Y:
        back[Y+Y:Y+h2+Y,X+X:X+X+w2] = fore
        #-x,+yではみ出す，
    elif -x >=X and y>=Y:
        back[Y+Y:Y+h2+Y,0:w2] = fore
        #+x,+yではみ出す，
    elif x >=X and y>=Y:
        back[Y:Y+h2+Y,X+X:X+X+w2] = fore
        #+x,-yではみ出す，
    elif x >=X and -y>=Y:
        back[0:h2,X+X:X+X+w2] = fore

"""
ここまで関数定義
"""

#画像の数だけbackが必要
back1 = cv2.imread("back.png",1)
back2 = cv2.imread("back.png",1)
back3 = cv2.imread("back.png",1)
back4 = cv2.imread("back.png",1)

calibration_img = cv2.imread("calibration.png",1)


cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)


button_pin1 = 17 # 11番端子
button_pin2 = 27 # 13番端子
button_pin3 = 22 # 15番端子


# GPIO初期化
wiringpi.wiringPiSetupGpio()
# GPIOを出力モード（1）に設定
wiringpi.pinMode( button_pin1, 0 )
wiringpi.pinMode( button_pin2, 0 )
wiringpi.pinMode( button_pin3, 0 )
# 端子に何も接続されていない場合の状態を設定
# 3.3Vの場合には「2」（プルアップ）
# 0Vの場合は「1」と設定する（プルダウン）
wiringpi.pullUpDnControl( button_pin1, 2 )
wiringpi.pullUpDnControl( button_pin2, 2 )
wiringpi.pullUpDnControl( button_pin3, 2 )

#↓ラズパイ(opencv2)の方でやらないとなぜか動かない(PCはopencv3)
# cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)




capture = cv2.VideoCapture(0)

count = 0
l = []
m = []
M = []
# isOpenedの代わりにTrueを使うと，frameがemptyのときエラーを吐く
while capture.isOpened():
    ret, frame = capture.read()
    if ret :
        #とりあえず常に黄色の輪郭をとってくるようにする．
        areas,res,_= getTarget(frame)
        cv2.imshow('frame',frame)
        #キャリブレーションボタンが押された場合，射影変換行列を計算
        if #wiringpi.digitalRead(button_pin) == 0:
            cv2.imshow("img",)
            time.sleep(3)
            areas,_,_=getBlue(frame)
            M = calibration(areas)

        #スイッチ1が押された場合．
        if wiringpi.digitalRead(button_pin1) == 0:
            #キャリブレーションボタンを押していた場合，画像を台形補正．
            if　M:
                img1 = revision(M,img1)
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
                    #輪郭を書き込む
                    cv2.drawContours(res, areas, -1, (0,0,255), 3)
                    #webcamera輪郭の重心計算
                    x, y = center_of_image(frame)
                    if not x == 0:
                        #トラッキング部分．重心の移動差を利用
                        l.append(x)
                        m.append(y)
                        count +=1
                        if count>4:
                            x_diff = l[count-1]-l[4]
                            y_diff = m[count-1]-m[4]
                            m1,n1 = frame.shape[:2]
                            m2,n2 = back.shape[:2]

                            #背景をリセットしてからオーバーレイ
                            back1 = cv2.imread("back1.png",1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back1,img1)
                            cv2.circle(res, (x,y), 10, (0, 0, 255), -1)
            cv2.imshow('img',back1)

        #スイッチ2が押された場合．
        elif wiringpi.digitalRead(button_pin2) == 0:
            #キャリブレーションボタンを押していた場合，画像を台形補正．
            if　M:
                img2 = revision(M,img2)
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
                    #輪郭を書き込む
                    cv2.drawContours(res, areas, -1, (0,0,255), 3)
                    #webcamera輪郭の重心計算
                    x, y = center_of_image(frame)
                    if not x == 0:
                        #トラッキング部分．重心の移動差を利用
                        l.append(x)
                        m.append(y)
                        count +=1
                        if count>4:
                            x_diff = l[count-1]-l[4]
                            y_diff = m[count-1]-m[4]
                            m1,n1 = frame.shape[:2]
                            m2,n2 = back.shape[:2]

                            #背景をリセットしてからオーバーレイ
                            back2 = cv2.imread("back2.png",1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back2,img2)
                            cv2.circle(res, (x,y), 10, (0, 0, 255), -1)
            cv2.imshow('img',back2)

        #スイッチ3が押された場合．
        elif wiringpi.digitalRead(button_pin3) == 0:
            #キャリブレーションボタンを押していた場合，画像を台形補正．
            if　M:
                img3 = revision(M,img3)
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
                    #輪郭を書き込む
                    cv2.drawContours(res, areas, -1, (0,0,255), 3)
                    #webcamera輪郭の重心計算
                    x, y = center_of_image(frame)
                    if not x == 0:
                        #トラッキング部分．重心の移動差を利用
                        l.append(x)
                        m.append(y)
                        count +=1
                        if count>4:
                            x_diff = l[count-1]-l[4]
                            y_diff = m[count-1]-m[4]
                            m1,n1 = frame.shape[:2]
                            m2,n2 = back3.shape[:2]

                            #背景をリセットしてからオーバーレイ
                            back3 = cv2.imread("back3.png",1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back3,img3)
                            cv2.circle(res, (x,y), 10, (0, 0, 255), -1)
            cv2.imshow('img',back3)

        #スイッチ4が押された場合．
        elif wiringpi.digitalRead(button_pin4) == 0:
            #キャリブレーションボタンを押していた場合，画像を台形補正．
            if　M:
                img4 = revision(M,img4)
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
                    #輪郭を書き込む
                    cv2.drawContours(res, areas, -1, (0,0,255), 3)
                    #webcamera輪郭の重心計算
                    x, y = center_of_image(frame)
                    if not x == 0:
                        #トラッキング部分．重心の移動差を利用
                        l.append(x)
                        m.append(y)
                        count +=1
                        if count>4:
                            x_diff = l[count-1]-l[4]
                            y_diff = m[count-1]-m[4]
                            m1,n1 = frame.shape[:2]
                            m2,n2 = back.shape[:2]

                            #背景をリセットしてからオーバーレイ
                            back4 = cv2.imread("back4.png",1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back4,img4)
                            cv2.circle(res, (x,y), 10, (0, 0, 255), -1)
            cv2.imshow('img',back4)

        #スイッチOFFが押された場合．
        else:
            cv2.imshow('img',back1)

        cv2.waitKey(-1)


#waitKeyの引数を0以下にするとキー入力する毎に画面がframeが更新する．
    if cv2.waitKey(-1) &  0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()