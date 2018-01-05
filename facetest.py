#coding: UTF-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_rect_of_target_color(image):
    #convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    #Extract red parts from image
    mask[((h < 20) | (h > 200)) & (s > 128)] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        epsilon = 0.15*cv2.arcLength(contours,True)
        approx = cv2.approxPolyDP(contours,epsilon,True)
        rects.append(np.array(approx))
    return rects


capture = cv2.VideoCapture(0)

if capture.isOpened() is False:
    raise("IO Error")

cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

while True:
    #retはframe取得が成功したか否かのフラグ
    ret, frame = capture.read()

    if ret == False:
        continue

    cv2.imshow("Capture", frame)

    rows,cols = frame.shape[:2]
    #img = cv2.resize(img,(cols/2, rows/2))
    # Convert BGR to HSV        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    image2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contoursの中身は輪郭が大きい方から順に格納されている．
    #print (contours[1])．．．点が5つ以上プリントされるので，頂点４つで格納しているのではない．

    #contoursを矩形近似
    areas=[]
    epsilon = 0.15*cv2.arcLength(contours[1],True)
    approx = cv2.approxPolyDP(contours[1],epsilon,True)
    areas.append(approx)
    print (approx)
    #矩形近似の時点で頂点4つで格納．左上，右上，右下，左下の順．
    img = cv2.drawContours(img, areas, -1, (0,255,0), 3)
    if cv2.waitKey(33) >= 0:
        cv2.imwrite("image.png", image)
        break

    cv2.destroyAllWindows()
