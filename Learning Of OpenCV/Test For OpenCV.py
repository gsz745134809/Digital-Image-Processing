# import cv2

# img = cv2.imread("D:\\cat.jpg")  # 原始
# # img = cv2.imread("D:\\cat.jpg", IMREAD_GRAYSCALE)  # 读入彩色图像
# # img = cv2.imread("D:\\cat.jpg", cv2.IMREAD_GRAYSCALE)  # 以灰度模式读入图像

# cv2.namedWindow("Image")  # 原始
# # cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)  # 命名窗口
# # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 命名窗口，可以调整窗口大小

# cv2.imshow("Image", img)  # 显示图像
# cv2.waitKey (0)  # 等待键盘输入，为毫秒级
# cv2.destroyAllWindows()  # 可以轻易删除任何我们建立的窗口，括号内输入想删除的窗口名


# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('D:\\cat.jpg', 0)
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])
# plt.show()

import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while(True):
        # captur
        e frame-by-frame

    ret, frame = cap.read()

    # our operation on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break
# when everything done , release the capture
cap.release()
cv2.destroyAllWindows()

