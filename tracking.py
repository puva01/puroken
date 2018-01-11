# -*- coding: utf-8 -*-
import cv2
#backにforeを中心から(x,y)移動させてオーバーレイ
def clip_image(x, y):
    global back
    w1, h1, _ = back.shape
    w2, h2, _ = fore.shape
    X = int((w1-w2)/2)
    Y = int((h1-h2)/2)
    back[X+x:X+w2+x, Y+y:Y+h2+y] = fore
#四角輪郭を取り出して，重心を計算
def center_of_image(image)
    _,_,contours = getBlue(image)
    cnt = contours[0]
    M = cv2.moments(cnt)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])

fore = cv2.imread("testimg.png")
back = cv2.imread("back.png")

#左からxピクセル,上からyピクセルのところに描画
clip_image(0, 0)

cv2.imshow("result", back)
cv2.waitKey(-1)

"""
backからforeがはみだすときうまくいかない
"""
