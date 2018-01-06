import cv2
import numpy as np

# areas = [[[732, 137]],[[374, 157]],[[400, 447]],[[739, 424]]]



areas = [[[[791, 375]],

       [[429, 371]],

       [[437, 657]],

       [[790, 553]]], "dtype=int32"]

pts1 = np.float32([[120,36],[280,36],[120,190],[280,190]])

k = (pts1[3][0]-pts1[2][0])/(areas[0][3][0][0]-areas[0][2][0][0])
h1 = (areas[0][1][0][1]-areas[0][2][0][1])*k
h2 = (areas[0][0][0][1]-areas[0][3][0][1])*k
delta1 = (areas[0][2][0][0]-areas[0][1][0][0])*k
delta2 = (areas[0][0][0][0]-areas[0][3][0][0])*k
x1 = pts1[2]+[-delta1,-h1]
x2 = pts1[3]+[delta2,-h2]
pts2 = np.float32(x1,x2,pts1[2],pts1[3])
print x1




print len(areas[0][0])

pts1 = np.float32([[120,36],[280,36],[120,190],[280,190]])


#k = (pts1[3][0]-pts1[2][0])/(areas[3][0][0]-areas[2][0][0])
# print k
