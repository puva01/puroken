#coding: UTF-8
import cv2

# Include Camera

if __name__=="__main__":

    capture = cv2.VideoCapture(0)

    if capture.isOpened() is False:
        raise("IO Error")

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

    while True:

        ret, image = capture.read()

        if ret == False:
            continue

        cv2.imshow("Capture", image)

        if cv2.waitKey(33) >= 0:
            cv2.imwrite("image.png", image)
            break

    cv2.destroyAllWindows()
