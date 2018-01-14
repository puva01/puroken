#coding: UTF-8
# GPIOを制御するライブラリ
import wiringpi
# タイマーのライブラリ
import time

# ボタンを繋いだGPIOの端子番号
button_pin = 17 # 11番端子

# GPIO初期化
wiringpi.wiringPiSetupGpio()
# GPIOを出力モード（1）に設定
wiringpi.pinMode( button_pin, 0 )
# 端子に何も接続されていない場合の状態を設定
# 3.3Vの場合には「2」（プルアップ）
# 0Vの場合は「1」と設定する（プルダウン）
wiringpi.pullUpDnControl( button_pin, 2 )
img = cv2.imread("blue.png")
img1 = cv2.imread("calibration.png")

# whileの処理は字下げをするとループの範囲になる（らしい）
while True:
    # GPIO端子の状態を読み込む
    # ボタンを押すと「0」、放すと「1」になる
    # GPIOの状態が0V(0)であるか比較
    if( wiringpi.digitalRead(button_pin) == 0 ):
        # 0V(0)の場合に表示
        print ("Switch ON")
        img = img1
        # cv2.imshow("img",img)
    else:
        # 3.3V(1)の場合に表示
        print ("Switch OFF")
        cv2.imshow("img",img)
        cv2.waitKey(100)
    time.sleep(0.05)
