#coding: UTF-8
#include "windows.h"
import cv2
import numpy as np
import time
from operator import itemgetter

import sys

capture = cv2.VideoCapture(0)

count = 0
l = []
m = []
# isOpenedの代わりにTrueを使うと，frameがemptyのときエラーを吐く
while capture.isOpened():
    ret, frame = capture.read()

    if ret :
        cv2.imshow('frame',frame)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
