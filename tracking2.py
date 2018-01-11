# -*- coding: utf-8 -*-
import cv2
import numpy as np

fore = cv2.imread('testimg.png', -1)  # -1を付けることでアルファチャンネルも読んでくれるらしい。
back = cv2.imread('back.png')

width, height = fore.shape[:2]

mask = fore[:,:,3]  # アルファチャンネルだけ抜き出す。
mask = cv2.cvtColor(mask, cv2.cv.CV_GRAY2BGR)  # 3色分に増やす。
mask = mask / 255.0  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。

fore = fore[:,:,:3]  # アルファチャンネルは取り出しちゃったのでもういらない。

back[0:height:, 0:width] *= 1 - mask  # 透過率に応じて元の画像を暗くする。
back[0:height:, 0:width] += fore * mask  # 貼り付ける方の画像に透過率をかけて加算。

cv2.imshow('out.jpg', back)
