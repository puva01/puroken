# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.png")
rows,cols = img.shape[:2]
img = cv2.resize(img,(cols/2, rows/2))
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
smooth=cv2.GaussianBlur(imgray,(11,11),0)
ret,thresh = cv2.threshold(smooth,155,255,0)
emage, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img_contours= np.copy(img)

min_area=10000
max_area=100000
#object_contours= [cnt for cnt in contours if max_area>cv2.contourArea(cnt) > min_area]
areas = []
#object_contours= [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

#perimeter = cv2.arcLength(object_contours[0],True)
epsilon = 0.1*cv2.arcLength(object_contours[0],True)
approx = cv2.approxPolyDP(object_contours[0],epsilon,True)
areas.append(approx)
cv2.drawContours(img_contours,areas, -1, (0,255,0), 3)
print(approx)
pts1 = np.float32(areas[0])
pts2 = np.float32([[0,0],[0,442],[442,442],[442,0]])

M = cv2.getPerspectiveTransform(pts1,pts2)
inv_M=np.linalg.inv(M)
dst = cv2.warpPerspective(img,M,(1000,600))

cv2.imshow("img",img_contours)
cv2.imshow("img2",dst)
cv2.waitKey(-1)

#plt.subplot(121),plt.imshow(img),plt.title('Input')
#plt.subplot(122),plt.imshow(dst),plt.title('Output')
#plt.show()
