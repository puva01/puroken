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
    pts1 = np.float32([[135,65],[400,65],[135,330],[400,330]]) #areasと同じ順　(左上，右上，左下，右下)
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
    x1 = pts1[2]+[-delta,-h]
    x2 = pts1[3]+[delta,-h]
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
    res = cv2.bitwise_and(frame,frame, mask= mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    res = cv2.bitwise_and(frame,frame, mask= mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        try:
            M = cv2.moments(areas[0])
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            print ("zerodivision")
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

#無地
back1 = cv2.imread("back.png",1)
back2 = cv2.imread("back.png",1)
back3 = cv2.imread("back.png",1)
back4 = cv2.imread("back.png",1)


img_right1 = cv2.imread("right1.png",1)
img_right2 = cv2.imread("right2.png",1)
img_right3 = cv2.imread("right3.png",1)
img_right4 = cv2.imread("right4.png",1)
img_left1 = cv2.imread("left1.png",1)
img_left2 = cv2.imread("left2.png",1)
img_left3 = cv2.imread("left3.png",1)
img_left4 = cv2.imread("left4.png",1)
img_stop1 = cv2.imread("stop1.png",1)
img_stop2 = cv2.imread("stop2.png",1)
img_stop3 = cv2.imread("stop3.png",1)
img_rest1 = cv2.imread("rest1.png",1)
img_rest2 = cv2.imread("rest2.png",1)


calibration_img = cv2.imread("calibration.png",1)
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
# cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)


button_pin1 = 4 # 7番端子
button_pin2 = 17 # 11番端子
button_pin3 = 27 # 13番端子
button_pin4 = 22 # 15番端子
button_pin5 = 10 # 19番端子



# GPIO初期化
wiringpi.wiringPiSetupGpio()
# GPIOを出力モード（1）に設定
wiringpi.pinMode( button_pin1, 0 )
wiringpi.pinMode( button_pin2, 0 )
wiringpi.pinMode( button_pin3, 0 )
wiringpi.pinMode( button_pin4, 0 )
wiringpi.pinMode( button_pin5, 0 )
# 端子に何も接続されていない場合の状態を設定
# 3.3Vの場合には「2」（プルアップ）
# 0Vの場合は「1」と設定する（プルダウン）
wiringpi.pullUpDnControl( button_pin1, 2 )
wiringpi.pullUpDnControl( button_pin2, 2 )
wiringpi.pullUpDnControl( button_pin3, 2 )
wiringpi.pullUpDnControl( button_pin4, 2 )
wiringpi.pullUpDnControl( button_pin5, 2 )


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
        #キャリブレーションボタンが押された場合(右)3秒間，射影変換行列を計算
        if wiringpi.digitalRead(button_pin1) == 0 :
            print ("calibrating...")
            #3秒間投影し，Mを計算
            while True:
                cv2.imshow('img',calibration_img)
                cv2.waitKey(1)
                areas,_,_=getBlue(frame)
                print "areas:{}".format(areas)
                if areas != [] and len(areas[0])==4:
                    M = calibration(areas)
                    print "M:{}".format(M)
                count = count+1
                print count
                time.sleep(0.5)
                if count>5:
                    if not M =[]:

                        img_right1 = cv2.imread("right1.png",1)
                        img_right2 = cv2.imread("right2.png",1)
                        img_right3 = cv2.imread("right3.png",1)
                        img_right4 = cv2.imread("right4.png",1)
                        img_left1 = cv2.imread("left1.png",1)
                        img_left2 = cv2.imread("left2.png",1)
                        img_left3 = cv2.imread("left3.png",1)
                        img_left4 = cv2.imread("left4.png",1)
                        img_stop1 = cv2.imread("stop1.png",1)
                        img_stop2 = cv2.imread("stop2.png",1)
                        img_stop3 = cv2.imread("stop3.png",1)
                        img_rest1 = cv2.imread("rest1.png",1)
                        img_rest2 = cv2.imread("rest2.png",1)

                        img_right1 = revision(M,img_right1)
                        img_right2 = revision(M,img_right2)
                        img_right3 = revision(M,img_right3)
                        img_right4 = revision(M,img_right4)
                        img_left1 = revision(M,img_left1)
                        img_left2 = revision(M,img_left2)
                        img_left3 = revision(M,img_left3)
                        img_left4 = revision(M,img_left4)
                        img_stop1 = revision(M,img_stop1)
                        img_stop2 = revision(M,img_stop2)
                        img_stop3 = revision(M,img_stop3)
                        img_rest1 = revision(M,img_rest1)
                        img_rest2 = revision(M,img_rest2)
                        print ("Calibration Finished")
                    else:
                        print ("Calibration Failed")
                    count = 0
                    break

        #スイッチ2(右)が押された場合．
        if wiringpi.digitalRead(button_pin2) == 0:
            print ("turnright")
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
                    #webcamera輪郭の重心計算
                    x, y = center_of_image(frame)
                    if not x == 0:
                        #トラッキング部分．重心の移動差を利用
                        l.append(x)
                        m.append(y)
                        count +=1
                        if count>1:
                            x_diff = l[count-1]-l[1]
                            y_diff = m[count-1]-m[1]
                            m1,n1 = frame.shape[:2]
                            m2,n2 = back.shape[:2]
                            #背景をリセットしてからオーバーレイ
                            back1 = cv2.imread("back.png",1)
                            back2 = cv2.imread("back.png",1)
                            back3 = cv2.imread("back.png",1)
                            back4 = cv2.imread("back.png",1)

                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back1,img_right1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back2,img_right2)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back3,img_right3)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back4,img_right4)

            cv2.imshow('img',back1)
            cv2.waitKey(500)
            cv2.imshow('img',back2)
            cv2.waitKey(500)
            cv2.imshow('img',back3)
            cv2.waitKey(500)
            cv2.imshow('img',back4)
            cv2.waitKey(800)

        #スイッチ3(左)が押された場合．
        if wiringpi.digitalRead(button_pin3) == 0:
            print ("turnright")
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
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
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back1,img_left1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back2,img_left2)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back3,img_left3)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back4,img_left4)

            cv2.imshow('img',back1)
            cv2.waitKey(500)
            cv2.imshow('img',back2)
            cv2.waitKey(500)
            cv2.imshow('img',back3)
            cv2.waitKey(500)
            cv2.imshow('img',back4)
            cv2.waitKey(800)

        #スイッチ4(stop)が押された場合．
        elif wiringpi.digitalRead(button_pin4) == 0:
            print ("turnright")
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
                    #webcamera輪郭の重心計算
                    x, y = center_of_image(frame)
                    if not x == 0:
                        #トラッキング部分．重心の移動差を利用
                        l.append(x)
                        m.append(y)
                        count +=1
                        if count>1:
                            x_diff = l[count-1]-l[1]
                            y_diff = m[count-1]-m[1]
                            m1,n1 = frame.shape[:2]
                            m2,n2 = back.shape[:2]
                            #背景をリセットしてからオーバーレイ
                            back1 = cv2.imread("back.png",1)
                            back2 = cv2.imread("back.png",1)
                            back3 = cv2.imread("back.png",1)

                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back1,img_stop1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back2,img_stop2)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back3,img_stop3)


            cv2.imshow('img',back1)
            cv2.waitKey(500)
            cv2.imshow('img',back2)
            cv2.waitKey(500)
            cv2.imshow('img',back3)
            cv2.waitKey(500)


        #スイッチ5(rest)が押された場合．
        elif wiringpi.digitalRead(button_pin5) == 0:
            print ("rest")
            #frameから輪郭をとる
            if areas:
                if len(areas[0])==4 :
                    #webcamera輪郭の重心計算
                    x, y = center_of_image(frame)
                    if not x == 0:
                        #トラッキング部分．重心の移動差を利用
                        l.append(x)
                        m.append(y)
                        count +=1
                        if count>1:
                            x_diff = l[count-1]-l[1]
                            y_diff = m[count-1]-m[1]
                            m1,n1 = frame.shape[:2]
                            m2,n2 = back.shape[:2]
                            #背景をリセットしてからオーバーレイ
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back1,img_rest1)
                            clip_image(x_diff*m2/m1,y_diff*m2/m1,back2,img_rest2)

            cv2.imshow('img',back1)
            cv2.waitKey(200)
            cv2.imshow('img',back2)
            cv2.waitKey(2000)


        #スイッチOFFのとき．
        else wiringpi.digitalRead(button_pin2) == 0:
            print ("switch off")
            back = cv2.imread("back.png",1)
            cv2.imshow('img',back)


#waitKeyの引数を0以下にするとキー入力する毎に画面がframeが更新する．
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
