#coding: UTF-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def revision(pts1,pts2,img):
    M = cv2.getPerspectiveTransform(pts1,pts2)
    inv_M =np.linalg.inv(M)
    dst = cv2.warpPerspective(img,inv_M,(600,400))
    return dst

# def revision2(img):
#     h,w=img.shape[:2]
#     h = int(h/2)
#     w = int(w/2)
#     #dは画像サイズ比でやったほうがいいっぽい
#     d = 15
#     pts1 = np.float32([[w-d,h-d],[w+d,h-d],[w-d,h+d],[w+d,h+d]])
#     pts2 = np.float32([[w-d-5,h-d-15],[w+d+5,h-d-15],[w-d,h+d],[w+d,h+d]])
#     M = cv2.getPerspectiveTransform(pts1,pts2)
#     print pts1
#     print pts2
#
#     inv_M =np.linalg.inv(M)
#     dst = cv2.warpPerspective(img,inv_M,(500,400))
#     return dst pts1, pts2


#オリジナル画像の中心付近の四角形
pts0 = np.float32([[125,75],[160,75],[125,114],[160,114]])
#x方向に±20移動してみる
pts1 = np.float32([[125+20,75],[160+20,75],[125+20,114],[160+20,114]])
pts2 = np.float32([[125-20,75],[160-20,75],[125-20,114],[160-20,114]])
#拡大してみる
pts3 = np.float32([[125-5,75-5],[160+5,75-5],[125-5,114+5],[160+5,114+5]])
#台形
pts4 = np.float32([[120,60],[165,60],[125,114],[160,114]])

#四角形のサイズを小さくし，移動させてみる
pts1 = np.float32([[50,50],[70,50],[50,70],[70,70]])
pts2 = np.float32([[45,45],[75,45],[50,70],[70,70]])

img = cv2.imread ("testimg.png",1)
img2 = cv2.imread ("blue.png",1)


dst = revision(pts1,pts2,img)
dst0 = revision(pts0,pts1,img)
dst1 = revision(pts0,pts2,img)
dst2 = revision(pts0,pts3,img)
dst3 = revision(pts0,pts4,img)

# dst5,pts1,pts2 = revision2(img1)
# dst6,pts3,pts4 = revision2(img2)

# img = cv.rectangle(img, (pts1[],   50), (50,  150), (255, 0, 0), 3, 4)
plt.subplot(231),plt.imshow(img),plt.title('Input')
plt.subplot(232),plt.imshow(dst0),plt.title('Output0')
plt.subplot(233),plt.imshow(dst1),plt.title('Output1')
plt.subplot(234),plt.imshow(dst2),plt.title('Output2')
plt.subplot(235),plt.imshow(dst3),plt.title('Output3')
plt.subplot(236),plt.imshow(dst),plt.title('Output4')
plt.show()
# cv2.imshow("img1",dst3)
# cv2.imshow("img2",dst4)
# cv2.imshow("img3",dst5)
# cv2.imshow("img4",dst6)
# cv2.waitKey(-1)
'''
結果
･射影変換は，移動先(pts2)に向かって平行移動する．したがって，射影行列の逆行列を使うと，pts2の平行移動と逆方向に平行移動する
･拡大に対しては拡大(逆行列では縮小)．
･画像中の部分的な4点の射影変換に対し，画像全体が歪み修正される．
結論
.pts1,pts2の面積および位置関係は大体同様にする必要がある．
'''
