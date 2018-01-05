#coding: UTF-8
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

count = 0
while count<100:

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    areas.append(np.array(approx))
    #for contour in contours:
    #    epsilon = 0.08*cv2.arcLength(contour,True)
    #    approx = cv2.approxPolyDP(contour,epsilon,True)
    #    areas.append(np.array(approx))

    cv2.drawContours(res, areas, -1, (0,0,255), 3)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    count +=1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
