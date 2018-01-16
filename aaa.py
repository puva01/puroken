#coding: UTF-8
#include "windows.h"
import cv2
import numpy as np
import time
from operator import itemgetter

#オリジナル画像に対応した台形座標取得
# def getPts(areas):
#     #ブルーの元画像の頂点．findContoursで見つけてくる．
#     img_area,_,_ = getTarget(img)
#     x00=img_area[0][0][0][0]
#     y00=img_area[0][0][0][1]
#     x11=img_area[0][1][0][0]
#     y11=img_area[0][1][0][1]
#     x22=img_area[0][2][0][0]
#     y22=img_area[0][2][0][1]
#     x33=img_area[0][3][0][0]
#     y33=img_area[0][3][0][1]
#     p00 = [x00,y00]#左上
#     p11 = [x11,y11]#左下
#     p22 = [x22,y22]#右下
#     p33 = [x33,y33]#右上
#     pts1 = np.float32([p00,p33,p11,p22]) #areasと同じ順　(左上，右上，左下，右下)
#     #areasの格納が普通の配列じゃないので，配列に書き直す．
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
#     #areasの4つの座標を左上，右上，左下，右下の順で並び替える．台形の場合x座標の順で一意に決定
#     #np.float32リストに対してitemgetterは使えないらしい
#     area.sort(key=itemgetter(0))
#     #スケール変換変換
#     k = (pts1[3][0]-pts1[2][0])/(area[3][0]-area[1][0])
#     h1 = (area[0][1]-area[1][1])*k
#     h2 = (area[3][1]-area[2][1])*k
#     delta1 = (area[0][0]-area[1][0])*k
#     delta2 = (area[2][0]-area[3][0])*k
#     x1 = pts1[2]+[-delta1,-h1]
#     x2 = pts1[3]+[delta2,-h2]
#     pts2 = np.float32([x1,x2,pts1[2],pts1[3]])
#     return pts1,pts2,x1,x2,k,delta1,h1,h2,area

def calibration(M,areas):
    pts1 = np.float32([[446,171],[790,173],[446,514],[788,514]]) #areasと同じ順　(左上，右上，左下，右下)
    #areasの格納が普通の配列じゃないので，配列に書き直す．
    x0=areas[0][0][0][0]
    y0=areas[0][0][0][1]
    x1=areas[0][1][0][0]
    y1=areas[0][1][0][1]
    x2=areas[0][2][0][0]
    y2=areas[0][2][0][1]
    x3=areas[0][3][0][0]
    y3=areas[0][3][0][1]
    p0 = [x0,y0]
    p1 = [x1,y1]
    p2 = [x2,y2]
    p3 = [x3,y3]
    area = [p0,p1,p2,p3]
    #areasの4つの座標を左上，右上，左下，右下の順で並び替える．台形の場合x座標の順で一意に決定
    #np.float32リストに対してitemgetterは使えないらしい
    area.sort(key=itemgetter(0))
    #スケール変換変換
    k = (pts1[3][0]-pts1[2][0])/(area[2][0]-area[1][0])
    h = (area[1][1]-area[0][1])*k
    delta = (area[1][0]-area[0][0])*k
    x1 = pts1[2]+[-delta,-h1]
    x2 = pts1[3]+[delta,-h2]
    pts2 = np.float32([x1,x2,pts1[2],pts1[3]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return M

#台形補正画像を生成する関数
def revision(M,img):
    inv_M =np.linalg.inv(M)
    dst = cv2.warpPerspective(img,inv_M,(0,800))
    return dst

#ターゲット(黄色)の輪郭を取ってくる関数
def getTarget(image):
    # Convert BGR to HSV and smooth
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    smooth=cv2.GaussianBlur(hsv,(15,15),0)
    #黄色
    lower_blue = np.array([25,100,85])
    upper_blue = np.array([100,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(smooth, lower_blue, upper_blue)

    # Bitwise-AND mask and original image(白黒画像の中で，白の部分だけ筒抜けになって映る)
    res = cv2.bitwise_and(image,image, mask= mask)
    image,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    contours.sort(key=cv2.contourArea,reverse=True)
    if  contours:
        cnt = contours[0]
        epsilon = 0.08*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        areas.append(approx)
    else:
        cnt = []
    return areas,res,cnt

#台形補正用にwebcameraから輪郭をとってくる．
def getBlue(frame):
    # Convert BGR to HSV and smooth
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    smooth=cv2.GaussianBlur(hsv,(15,15),0)

    # define range of blue color in HSV　(第1引数を110〜130→90〜140に変更)
    lower_blue = np.array([90,90,90])
    upper_blue = np.array([140,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(smooth, lower_blue, upper_blue)

    # Bitwise-AND mask and original image(白黒画像の中で，白の部分だけ筒抜けになって映る)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    image,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    contours.sort(key=cv2.contourArea,reverse=True)
    if  contours:
        cnt = contours[0]
        epsilon = 0.08*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        areas.append(approx)
    else:
        cnt = []
    return areas,res,cnt

#輪郭の重心を計算
def center_of_image(image):
    areas,_,_ = getTarget(image)
    if areas:
        global x ,y
        try :
            M = cv2.moments(areas[0])
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            print ("zero division")
    else:
        x=0
        y=0

    return x,y


#backをforeを中心から(x,y)移動させて重ね合わせる
# def clip_image(x, y, img):
#     global back
#     h1, w1, _ = back.shape
#     h2, w2, _ = img.shape
#     X = int((w1-w2)/2)
#     Y = int((h1-h2)/2)
#     if abs(x)< X and abs(y)<Y:
#         back[Y+y:Y+h2+y,X+x:X+w2+x] = img
#         #+xではみ出す，
#     elif x >=X and abs(y)<Y:
#         back[Y+y:Y+h2+y,X+X:X+w2+X,] = img
#         #-xではみ出す，
#     elif -x >=X and abs(y)<Y:
#         back[Y+y:Y+h2+y,0:w2] = img
#         #+yではみ出す，
#     elif abs(x)<X and y>=Y:
#         back[Y+Y:Y+h2+Y,X+x:X+w2+x] = img
#         #-yではみ出す，
#     elif abs(x)<X and -y>Y:
#         back[0:h2,X+x:X+w2+x] = img
#         #+x，+yではみ出す，
#     elif x >=X and y>=Y:
#         back[Y+Y:Y+h2+Y,X+X:X+X+w2] = img
#         #-x,+yではみ出す，
#     elif -x >=X and y>=Y:
#         back[Y+Y:Y+h2+Y,0:w2] = img
#         #+x,+yではみ出す，
#     elif x >=X and y>=Y:
#         back[Y:Y+h2+Y,X+X:X+X+w2] = img
#         #+x,-yではみ出す，
#     elif x >=X and -y>=Y:
#         back[0:h2,X+X:X+X+w2] = img

def clip_image(x, y, back, fore):
    h1, w1, _ = back.shape
    h2, w2, _ = fore.shape
    X = int((w1-w2)/2)
    Y = int((h1-h2)/2)
    if abs(x)< X and abs(y)<Y:
        back[Y+y:Y+h2+y,X+x:X+w2+x] = fore
        #+xではみ出す，
    elif x >=X and abs(y)<Y:
        back[Y+y:Y+h2+y,X+X:X+w2+X,] = fore
        #-xではみ出す，
    elif -x >=X and abs(y)<Y:
        back[Y+y:Y+h2+y,0:w2] = fore
        #+yではみ出す，
    elif abs(x)<X and y>=Y:
        back[Y+Y:Y+h2+Y,X+x:X+w2+x] = fore
        #-yではみ出す，
    elif abs(x)<X and -y>Y:
        back[0:h2,X+x:X+w2+x] = fore
        #+x，+yではみ出す，
    elif x >=X and y>=Y:
        back[Y+Y:Y+h2+Y,X+X:X+X+w2] = fore
        #-x,+yではみ出す，
    elif -x >=X and y>=Y:
        back[Y+Y:Y+h2+Y,0:w2] = fore
        #+x,+yではみ出す，
    elif x >=X and y>=Y:
        back[Y:Y+h2+Y,X+X:X+X+w2] = fore
        #+x,-yではみ出す，
    elif x >=X and -y>=Y:
        back[0:h2,X+X:X+X+w2] = fore



#imgに青い輪郭がないものを選ぶとエラーが出る．
#img = cv2.imread("bluerect2.png",1)
img_stop1 = cv2.imread("stop1.png",1)
img_stop2 = cv2.imread("stop2.png",1)
img_stop3 = cv2.imread("stop3.png",1)
img_rest1 = cv2.imread("rest1.png",1)
img_rest2 = cv2.imread("rest2.png",1)

back4 = cv2.imread("back.png",1)
back3 = cv2.imread("back.png",1)
back2 = cv2.imread("back.png",1)
back1 = cv2.imread("back.png",1)
back = cv2.imread("back.png",1)
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
#areas0,res0,_= getTarget(img1)
#cv2.drawContours(res0, areas0, -1, (0,0,255), 3)
#cv2.imshow("img0",img1)

#↓ラズパイ(opencv2)の方でやらないとなぜか動かない(PCはopencv3)
#cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)



capture = cv2.VideoCapture(0)
capture.set(3, 800)  # Width
capture.set(4, 800)  # Heigh
count = 0
l = []
m = []
# isOpenedの代わりにTrueを使うと，frameがemptyのときエラーを吐く
while capture.isOpened():
    ret, frame = capture.read()
    capture.set(3, 800)  # Width
    capture.set(4, 800)  # Heigh

    if ret :
        #frameから輪郭をとる
        areas,res,_= getTarget(frame)

        cv2.imshow('frame',frame)
        #print len(areas[0])
        #print areas

        if areas:
            if len(areas[0])==4 :
                #輪郭を書き込む
                cv2.drawContours(res, areas, -1, (0,0,255), 3)
                #webcamera輪郭の重心計算
                x, y = center_of_image(frame)
                if not x == 0:
                    #トラッキング部分．重心の移動差を利用
                    l.append(x)
                    m.append(y)
                    count +=1
                    if count>5:
                        x_diff = l[count-1]-l[5]
                        y_diff = m[count-1]-m[5]
                        m1,n1 = frame.shape[:2]
                        m2,n2 = back.shape[:2]


                        cv2.circle(res, (x,y), 10, (0, 0, 255), -1)
                        #frame上の重心をimg上の重心に変換
                        m1,n1 = frame.shape[:2]
                        m2,n2 = back.shape[:2]

                        X = int(x*m2/m1)
                        Y = int(y*n2/n1)
                        #取得した輪郭の重心を計算し，丸を描く
                        # cv2.circle(back, (X,Y), 10, (0, 0, 255), -1)
                        back1 = cv2.imread("back.png",1)
                        back2 = cv2.imread("back.png",1)
                        back3 = cv2.imread("back.png",1)
                        back4 = cv2.imread("back.png",1)

                        clip_image(x_diff*m2/m1,y_diff*m2/m1,back1,img_rest1)
                        clip_image(x_diff*m2/m1,y_diff*m2/m1,back2,img_rest2)


        cv2.imshow('img',back1)
        cv2.imshow('img',back2)





        cv2.imshow('res',res)


            #webカメラ上の輪郭を取得,台形補正画像生成
            # pts1,pts2,x1,x2,k,delta1,h1,h2,area = getPts(areas)
            # dst = revision(pts1,pts2,img)
            #cv2.imshow("dst",dst)

#waitKeyの引数を0以下にするとキー入力する毎に画面がframeが更新する．
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
