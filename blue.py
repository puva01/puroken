# -*- coding: utf-8 -*-
#include "windows.h"
import cv2
import time

img1 = cv2.imread("blue.png")

#フルスクリーン表示
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

cv2.imshow("img",img1)
cv2.waitKey(500)
