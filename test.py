#coding: UTF-8
import cv2
import numpy as np

#
# def find_rect_of_target_color(image):
#   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
#   h = hsv[:, :, 0]
#   s = hsv[:, :, 1]
#   mask = np.zeros(h.shape, dtype=np.uint8)
#   mask[((h < 20) | (h > 200)) & (s > 128)] = 255
#   contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#   rects = []
#   for contour in contours:
#     approx = cv2.convexHull(contour)
#     rect = cv2.boundingRect(approx)
#     rects.append(np.array(rect))
#   return rects

# キャリブレーション関数 引数:投影画像の4点
# def calibration(areas)
#     pts1 = np.float32()
#
#     x0=areas[0][0][0][0]
#     y0=areas[0][0][0][1]
#     x1=areas[0][1][0][0]
#     y1=areas[0][1][0][1]
#     x2=areas[0][2][0][0]
#     y2=areas[0][2][0][1]
#     x3=areas[0][3][0][0]
#     y3=areas[0][3][0][1]
#     p0 = [x0,y0]
#     p1 = [x1,y1]
#     p2 = [x2,y2]
#     p3 = [x3,y3]
#     area = [p0,p1,p2,p3]
#     pts2 =

def clip_image(x, y):
    global back
    h1, w1, _ = back.shape
    h2, w2, _ = img.shape
    X = int((w1-w2)/2)
    Y = int((h1-h2)/2)
    if abs(x)< X and abs(y)<Y:
        back[Y+y:Y+h2+y,X+x:X+w2+x] = img

img = cv2.imread("blue.png",1)
back = cv2.imread("back.png",1)

clip_image(0,0)
cv2.imshow("img",back)
cv2.waitKey(-1)
