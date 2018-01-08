#coding: UTF-8
#include "windows.h"
import cv2
import numpy as np
import time
from operator import itemgetter

#オリジナル画像に対応した台形座標取得
def getPts(areas):
    #ブルーの元画像の頂点．findContoursで見つけてくる．
    img_area,_,_ = getBlue(img)
    x00=img_area[0][0][0][0]
    y00=img_area[0][0][0][1]
    x11=img_area[0][1][0][0]
    y11=img_area[0][1][0][1]
    x22=img_area[0][2][0][0]
    y22=img_area[0][2][0][1]
    x33=img_area[0][3][0][0]
    y33=img_area[0][3][0][1]
    p00 = [x00,y00]#左上
    p11 = [x11,y11]#左下
    p22 = [x22,y22]#右下
    p33 = [x33,y33]#右上
    pts1 = np.float32([p00,p33,p11,p22]) #areasと同じ順　(左上，右上，左下，右下)
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
    dst = cv2.warpPerspective(img,inv_M,(800,500))
    return dst

#青の輪郭を取ってくる関数
def getBlue(image):
    # Convert BGR to HSV and smooth
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
    cnt = contours[0]
    epsilon = 0.08*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    # areas.append(np.array(approx))
    areas.append(approx)
    return areas,res,cnt

#輪郭の重心を計算
def center_of_image(image):
    _,_,cnt = getBlue(image)
    M = cv2.moments(cnt)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    return x,y


img = cv2.imread("bluerect2.png",1)
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
areas0,res0,_= getBlue(img)
cv2.drawContours(res0, areas0, -1, (0,0,255), 3)
cv2.imshow("img0",img)
#↓ラズパイ(opencv2)の方でやらないとなぜか動かない(PCはopencv3)
#cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)




capture = cv2.VideoCapture(0)

count = 0
# isOpenedの代わりにTrueを使うと，frameがemptyのときエラーを吐く
while capture.isOpened():
    ret, frame = capture.read()

    if ret :
        #frameから輪郭をとる
        areas,res,_= getBlue(frame)
        cv2.drawContours(res, areas, -1, (0,0,255), 3)
        x, y = center_of_image(frame)
        cv2.circle(res, (x,y), 10, (0, 0, 255), -1)
        cv2.imshow('frame',frame)
        cv2.imshow('res',res)
        if len(areas[0])==4 :
            pts1,pts2,x1,x2,k,delta1,h1,h2,area = getPts(areas)
            dst = revision(pts1,pts2,img)
            #frame上の重心をimg上の重心に変換
            m1,n1 = frame.shape[:2]
            m2,n2 = img.shape[:2]
            X = int(x*m2/m1)
            Y = int(y*n2/n1)
            #print m1,m2,X
            cv2.circle(dst, (X,Y), 10, (0, 0, 255), -1)
            cv2.imshow('img',dst)



#waitKeyの引数を0以下にするとキー入力する毎に画面がframeが更新する．
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
