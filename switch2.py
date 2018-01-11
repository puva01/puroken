#include "windows.h"
import cv2
import sys
import termios
import time
import wiringpi

img1 = cv2.imread("white.jpg")
img2 = cv2.imread("right2.jpg")
img3 = cv2.imread("turnright2.jpg")

#フルスクリーン表示
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCRE$


pin = 17
wiringpi.wiringPiSetupGpio()
wiringpi.pinMode( pin, 0 )
wiringpi.pullUpDnControl( button_pin, 2 )

while True:
 if wiringpi.digitalRead(pin) == 0 :
  while True:
   cv2.imshow('img',img1)
   cv2.waitKey(1000)
   cv2.imshow('img',img2)
   cv2.waitKey(1000)
   print ("Switch On")
   time.sleep(1)
 else :
  cv2.imshow('img',img3)
  cv2.waitKey(1000)
  print ("Switch Off")
  time.sleep(1)
