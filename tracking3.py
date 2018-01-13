# -*- coding: utf-8 -*-
import cv2

def clip_image(x, y):
    global back
    h1, w1, _ = back.shape
    h2, w2, _ = img.shape
    X = int((w1-w2)/2)
    Y = int((h1-h2)/2)
    if abs(x)< X and abs(y)<Y:
        back[Y+y:Y+h2+y,X+x:X+w2+x] = img
        #+xではみ出す，
    elif x >=X and abs(y)<Y:
        back[Y+y:Y+h2+y,X+X:X+w2+X,] = img
        #-xではみ出す，
    elif -x >=X and abs(y)<Y:
        back[Y+y:Y+h2+y,0:w2] = img
        #+yではみ出す，
    elif abs(x)<X and y>=Y:
        back[Y+Y:Y+h2+Y,X+x:X+w2+x] = img
        #-yではみ出す，
    elif abs(x)<X and -y>Y:
        back[0:h2,X+x:X+w2+x] = img
        #+x，+yではみ出す，
    elif x >=X and y>=Y:
        back[Y+Y:Y+h2+Y,X+X:X+X+w2] = img
        #-x,+yではみ出す，
    elif -x >=X and y>=Y:
        back[Y+Y:Y+h2+Y,0:w2] = img
        #+x,+yではみ出す，
    elif x >=X and y>=Y:
        back[Y:Y+h2+Y,X+X:X+X+w2] = img
        #+x,-yではみ出す，
    elif x >=X and -y>=Y:
        back[0:h2,X+X:X+X+w2] = img

img0 = cv2.imread("right2.jpg",1)
back = cv2.imread("back.png",1)
h,w = img0.shape[:2]
img = cv2.resize(img0, (w/ 5, h/5))
clip_image(30,100)
cv2.imshow("img",back)
cv2.waitKey(-1)
