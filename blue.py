# -*- coding: utf-8 -*-
#include "windows.h"
import cv2
import matplotlib.pyplot as plt
import numpy as np

def revision(pts1,pts2,img):
    M = cv2.getPerspectiveTransform(pts1,pts2)
    inv_M =np.linalg.inv(M)
    dst = cv2.warpPerspective(img,inv_M,(400,220))
    return dst

img1 = cv2.imread("blue2.png",1)
#img1 = cv2.resize(img,(400,200))
#フルスクリーン表示
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
pts1=np.float32([[76,61],[282,61],[119,178],[240,178]])
pts2=np.float32([[119,58],[240,58],[119,178],[240,178]])
dst = revision(pts1,pts2,img1)

plt.subplot(121),plt.imshow(img1),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output1')
plt.show()
#cv2.imshow("img",img1)
#cv2.waitKey(-1)
