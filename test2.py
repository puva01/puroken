# -*- coding: utf-8 -*-
import cv2
#画像の上に画像を置く関数、x,yで書き込む位置を指定する
def clip_image(x, y):
    global back
    h, w, _ = fore.shape
    X = int((w1-w2)/2)
    Y = int((h1-h2)/2)
    back[Y+y:Y+y+h, X+x:X+x+w] = fore
#前景画像の読み込み
fore = cv2.imread("blue.png")
#背景画像の読み込み
back = cv2.imread("back.png")
#左から400ピクセル上から50ピクセルのところに描画
clip_image(0, 0)
cv2.imshow("result", back)
cv2.waitKey(-1)
