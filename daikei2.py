#coding: UTF-8
import cv2
import numpy as np
import time

#オリジナル画像に対応した台形座標取得
def getPts(areas):
    pts1 = np.float32([[120,36],[280,36],[120,190],[280,190]])
    #スケール変換変換
    k = (pts1[3][0]-pts1[2][0])/(areas[0][3][0][0]-areas[0][2][0][0])
    h1 = (areas[0][1][0][1]-areas[0][2][0][1])*k
    h2 = (areas[0][0][0][1]-areas[0][3][0][1])*k
    delta1 = (areas[0][2][0][0]-areas[0][1][0][0])*k
    delta2 = (areas[0][0][0][0]-areas[0][3][0][0])*k
    x1 = pts1[2]+[-delta1,-h1]
    x2 = pts1[3]+[delta2,-h2]
    pts2 = np.float32([x1,x2,pts1[2],pts1[3]])
    return pts1,pts2

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
        time.sleep(2)
        if len(areas[0])==4 :
            pts1,pts2 = getPts(areas)
            dst = revision(pts1,pts2,img)
            break

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

capture.release()
cv2.imshow('img',dst)
cv2.waitKey(-1)

cv2.destroyAllWindows()

# cv2.imshow('img',img)
# cv2.waitKey(-1)
