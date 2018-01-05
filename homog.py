# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.png")
rows,cols = img.shape[:2]
#img = cv2.resize(img,(cols/2, rows/2))
#グレースケール，ガウシアン平滑化，2値化
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
smooth=cv2.GaussianBlur(imgray,(11,11),0)
ret,thresh = cv2.threshold(smooth,155,255,0)
#第4引数cv2.CHAIN_APPROX_SIMPLEは輪郭のうち格納する点が最も少ない．
emage, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#print (contours[1])．．．点が5つ以上プリントされるので，頂点４つで格納しているのではない．
#contoursの中身は輪郭が大きい方から順に格納されている．


areas=[]

epsilon = 0.15*cv2.arcLength(contours[1],True)
approx = cv2.approxPolyDP(contours[1],epsilon,True)
areas.append(approx)
print (approx)
#矩形近似の時点で頂点4つで格納．左上，右上，右下，左下の順．
img = cv2.drawContours(img, areas, -1, (0,255,0), 3)
#第二引数は-１でareasの全ての矩形を表示．今はcontours[1]で計算してるのでもともと1つだけ表示．
pts1=np.float32(areas[0])
pts2=np.float32([[100,50],[400,50],[360,220],[140,220]])
M = cv2.getPerspectiveTransform(pts1,pts2)
inv_M =np.linalg.inv(M)
dst = cv2.warpPerspective(img,inv_M,(500,280))
plt.subplot(131),plt.imshow(img),plt.title('Input')
plt.subplot(132),plt.imshow(dst),plt.title('Output')
plt.subplot(133),plt.imshow(emage),plt.title('Output')
plt.show()
