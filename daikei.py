#coding: UTF-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_rect_of_target_color(image):
    #convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    smooth=cv2.GaussianBlur(hsv,(11,11),0)
    mask = np.zeros(h.shape, dtype=np.uint8)
    #Extract red parts from image
    mask[((h < 20) | (h > 200)) & (s > 128)] = 255
    image,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for contour in contours:
        epsilon = 0.15*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        areas.append(np.array(approx))
    #epsilon = 0.15*cv2.arcLength(contours[0],True)
    #approx = cv2.approxPolyDP(contours[0],epsilon,True)
    #areas.append(np.array(approx))
    return areas

capture = cv2.VideoCapture(0)
while cv2.waitKey(30) < 0:
    _, frame = capture.read()
    areas = find_rect_of_target_color(frame)
    cv2.drawContours(frame, areas, -1, (0,0,255), 3)
    cv2.imshow('red', frame)
    cv2.imshow('red', mask)
capture.release()
cv2.destroyAllWindows()
