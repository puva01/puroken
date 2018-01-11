# -*- coding: utf-8 -*-
import cv2

def tracking(x,y,back,fore):
    height, width = src.shape[:2]  # サイズを取得しておく。
    # dst[上のy座標:下のy座標, 左のx座標:右のx座標]
    back[x:x+width, y:y+height] = fore


fore = cv2.imread('testimg.png')  #　前景
back = cv2.imread('back.png')  # 背景

tracking(30,30,fore,back)
cv2.imshow('img', back)
cv2.waitKey(-1)
