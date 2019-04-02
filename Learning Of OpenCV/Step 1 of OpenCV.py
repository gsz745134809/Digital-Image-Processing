# 图像就是一个矩阵，在OpenCV for Python中，图像就是NumPy中的数组！
# 如果读取图像首先要导入OpenCV包，方法为：
# import cv2

# 1.读取并显示图像
# img = cv2.imread("D:\\cat.jpg")
# img = cv2.imread("D:\\cat.jpg", cv2.IMREAD_COLOR)  # 读入一副彩色图像。图像的透明度会被忽略，这是默认参数
# img = cv2.imread("D:\\cat.jpg", cv2.IMREAD_GRAYSCALE)  # 以灰度模式读入图像  
# img = cv2.imread("D:\\cat.jpg", cv2.IMREAD_UNCHANGED)  # 读入一幅图像，并且包括图像的 alpha 通道 

# 接着创建一个窗口
# cv2.namedWindow("Image")  #原始
# cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)  # 命名窗口
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # 命名窗口，可以调整窗口大小

# 然后在窗口中显示图像
# cv2.imshow("Image", img)

# 最后还要添上一句：
# cv2.waitKey (0)
# 如果不添最后一句，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过。

# import cv2 
# 完整程序 
# img = cv2.imread("D:\\cat.jpg") 
# cv2.namedWindow("Image") 
# cv2.imshow("Image", img) 
# cv2.waitKey (0)

# cv2.destroyAllWindows()
# 最后释放窗口是个好习惯！


# 2.创建/复制图像
# import numpy as np

# 新的OpenCV的接口中没有CreateImage接口。即没有cv2.CreateImage这样的函数。
# 如果要创建图像，需要使用numpy的函数（现在使用OpenCV-Python绑定，numpy是必装的）。如下：
# emptyImage = np.zeros(img.shape, np.uint8)

# 在新的OpenCV-Python绑定中，图像使用NumPy数组的属性来表示图像的尺寸和通道信息。
# 如果输出img.shape，将得到(500, 375, 3)，这里是以OpenCV自带的cat.jpg为示例。最后的3表示这是一个RGB图像。

# 也可以复制原有的图像来获得一副新图像。
# emptyImage2 = img.copy()

# 如果不怕麻烦，还可以用cvtColor获得原图像的副本。
# emptyImage3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# emptyImage3[...]=0
# 后面的emptyImage3[...]=0是将其转成空白的黑色图像。


# 3.保存图像

# 保存图像很简单，直接用cv2.imwrite即可。
# cv2.imwrite("D:\\cat2.jpg", img)
# 第一个参数是保存的路径及文件名，第二个是图像矩阵。其中，imwrite()有个可选的第三个参数，如下：
# cv2.imwrite("D:\\cat2.jpg", img，[int(cv2.IMWRITE_JPEG_QUALITY), 5])
# 第三个参数针对特定的格式： 对于JPEG，其表示的是图像的质量，用0-100的整数表示，默认为95。 
# 注意，cv2.IMWRITE_JPEG_QUALITY类型为Long，必须转换成int。

# 对于PNG，第三个参数表示的是压缩级别。cv2.IMWRITE_PNG_COMPRESSION，从0到9,压缩级别越高，图像尺寸越小。默认级别为3：
# cv2.imwrite("./cat.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) 
# cv2.imwrite("./cat2.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])






import cv2
import numpy as np
 
img = cv2.imread("D:\\cat.jpg")  # 读取图像
emptyImage = np.zeros(img.shape, np.uint8)  
 
emptyImage2 = img.copy()

emptyImage3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#emptyImage3[...]=0
 
cv2.imshow("EmptyImage", emptyImage)  # 在窗口中显示图像
cv2.imshow("Image", img)  # 在窗口中显示图像
cv2.imshow("EmptyImage2", emptyImage2)  # 在窗口中显示图像
cv2.imshow("EmptyImage3", emptyImage3)  # 在窗口中显示图像
cv2.imwrite("D:\\cat2.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 5])  # 保存图像
cv2.imwrite("D:\\cat3.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 保存图像
cv2.imwrite("D:\\cat.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # 保存图像
cv2.imwrite("D:\\cat2.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])  # 保存图像
cv2.waitKey (0)
cv2.destroyAllWindows()



# 灰度转换
# 灰度转换的作用就是：转换成灰度的图片的计算强度得以降低。
import cv2
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

# 画图
# opencv 的强大之处的一个体现就是其可以对图片进行任意编辑，处理。 
# 下面的这个函数最后一个参数指定的就是画笔的大小。
import cv2
cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

















