import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 图像的几何变换

def ImageTranslation(filepath1):
    img = cv.imread(filepath1)
    height, width = img.shape[:2]
    print("height,width = ",height, width)

    dx, dy = 100, 50
    MAT = np.float32([[1,0,dx], [0,1,dy]])  # 构造平移变换矩阵
    
    imgTrans1 = cv.warpAffine(img,MAT, (width,height))
    imgTrans2 = cv.warpAffine(img,MAT, (601,401), borderValue=(0,0,0))

    plt.figure(figsize=(9,3.2))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. translation 1"),plt.imshow(cv.cvtColor(imgTrans1, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. translation 2"),plt.imshow(cv.cvtColor(imgTrans2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def ImageResize(filepath1):
    img = cv.imread(filepath1)
    height, width = img.shape[:2]
    print("height,width = ",height, width)
    
    imgResize1 = cv.resize(img, (600,480))
    imgResize2 = cv.resize(img, None, fx=1.2, fy=0.8, interpolation=cv.INTER_CUBIC)

    plt.figure(figsize=(9,3.2))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. resize 1"),plt.imshow(cv.cvtColor(imgResize1, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. resize 2"),plt.imshow(cv.cvtColor(imgResize2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def ImageRotate(filepath1):
    img = cv.imread(filepath1)
    height, width = img.shape[:2]
    print("height,width = ",height, width)

    # 以原点为中心旋转
    x0, y0 = 0, 0  # 左上顶点
    theta, scale = 30, 1.0 # 逆时针旋转30°， 缩放系数1.0
    MAR0 = cv.getRotationMatrix2D((x0,y0), theta, scale)  # 旋转变换矩阵
    imgRot1 = cv.warpAffine(img, MAR0, (width, height))

    # 以任意点为中心旋转
    x0,y0 = width/2, height/2 # 图像中心
    angle = theta * np.pi/180  #弧度->角度
    wRot = int(width*np.cos(angle) + height*np.sin(angle))  # 调整宽度
    hRot = int(height*np.cos(angle) + width*np.sin(angle))  # 调整高度
    scale = width/wRot
    MAR1 = cv.getRotationMatrix2D((x0,y0), theta, 1.0)
    MAR2 = cv.getRotationMatrix2D((x0,y0), theta, scale)

    imgRot2 = cv.warpAffine(img, MAR1, (height,width), borderValue=(255,255,255))
    imgRot3 = cv.warpAffine(img, MAR2, (height,width))
    print(img.shape, imgRot2.shape, imgRot3.shape, scale)

    # 图像直角旋转
    imgRot90 = cv.rotate(img,cv.ROTATE_90_CLOCKWISE)
    imgRot180 = cv.rotate(img,cv.ROTATE_180)
    imgRot270 = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)

    






if __name__ == "__main__" :
    filepath1 = "/home/calmcar/work/python/opencv_test/img/lenna.bmp"
    filepath2 = "/home/calmcar/work/python/opencv_test/img/monarch.bmp"

    # ImageTranslation(filepath1)
    # ImageResize(filepath1)
    ImageRotate(filepath1)