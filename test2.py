# -*- coding: utf-8 -*-
import cv2

img1 = cv2.imread("calibrating.png")
cv2.imshow("img1",img1)
resized_img = cv2.resize(img1,(1335, 750))
#cv2.resize(img1,(1335,750))
cv2.imshow("img",resized_img)
cv2.waitKey(-1)
