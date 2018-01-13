#coding: UTF-8
import cv2
import numpy as np



# キャリブレーション関数 引数:投影画像の4点 areas =getTarget(blue)(frame)
def calibration(img):
    pts1 = np.float32([[446,171],[790,173],[446,514],[788,514]])
    pts2 = np.float32([[400,155],[836,155],[446,514],[788,514]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    inv_M =np.linalg.inv(M)
    dst = cv2.warpPerspective(img,inv_M,(1335,750))
    return dst


img = cv2.imread("right2.jpg",1)

dst = calibration(img)
cv2.imshow("img",img)
cv2.imshow("img1",dst)
cv2.waitKey(-1)
