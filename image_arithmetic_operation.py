import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 图像的算数运算

def ImageAdd(filepath1, filepath2):
    img1 = cv.imread(filepath1)
    img2 = cv.imread(filepath2)

    h,w = img1.shape[:2]
    img3 = cv.resize(img2, (w,h))

    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    print(img1.shape, img2.shape, img3.shape, gray.shape)

    # 图像与常数相加
    value = 100
    imgAddV = cv.add(img1, value)
    imgAddG = cv.add(gray, value)

    # 彩色图像与标量相加
    scalar = (30,40,50,60)
    # scalar = np.ones((1,3))*value
    # scalar = np.array([[40,50,60]])
    imgAddS = cv.add(img1, scalar)

    # Numpy 取模加法
    imgAddP = img1+img3;

    # opnecv 饱和加法
    imgAddCV = cv.add(img1, img3)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. img1"),plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(232),plt.axis('off'),plt.title("2. img1 + value"),plt.imshow(cv.cvtColor(imgAddV, cv.COLOR_BGR2RGB))
    plt.subplot(233),plt.axis('off'),plt.title("3. img1 + scalar"),plt.imshow(cv.cvtColor(imgAddS, cv.COLOR_BGR2RGB))

    plt.subplot(234),plt.axis('off'),plt.title("4. img3"),plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.subplot(235),plt.axis('off'),plt.title("5. numpy  img1 + img3"),plt.imshow(cv.cvtColor(imgAddP, cv.COLOR_BGR2RGB))
    plt.subplot(236),plt.axis('off'),plt.title("6. opnecv img1 + img3"),plt.imshow(cv.cvtColor(imgAddCV, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


def ImageMask(filepath1, filepath2):
    img1 = cv.imread(filepath1)
    img2 = cv.imread(filepath2)

    h,w = img1.shape[:2]
    img3 = cv.resize(img2, (w,h))
    print(img1.shape, img2.shape, img3.shape)

    imgAddCV = cv.add(img1, img3)

    # 淹模加法，矩形淹模图像
    maskRec = np.zeros(img1.shape[:2], np.uint8) # 生辰黑色模版
    xmin, ymin, w, h = 179, 190, 200, 200 # 矩形roi
    maskRec[ymin: ymin+h, xmin: xmin+w] = 255
    imgAddRec = cv.add(img1, img3, mask = maskRec)

    maskCir = np.zeros(img1.shape[:2], np.uint8) # 生辰黑色模版
    cv.circle(maskCir, (280,280), 120, 255, -1)
    imgAddCir = cv.add(img1, img3, mask = maskCir)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. img1"),plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(232),plt.axis('off'),plt.title("2. rect mask"),plt.imshow(cv.cvtColor(maskRec, cv.COLOR_BGR2RGB))
    plt.subplot(233),plt.axis('off'),plt.title("3. mask add"),plt.imshow(cv.cvtColor(imgAddRec, cv.COLOR_BGR2RGB))

    plt.subplot(234),plt.axis('off'),plt.title("4. img3"),plt.imshow(cv.cvtColor(imgAddCV, cv.COLOR_BGR2RGB))
    plt.subplot(235),plt.axis('off'),plt.title("5. circle mask"),plt.imshow(cv.cvtColor(maskCir, cv.COLOR_BGR2RGB))
    plt.subplot(236),plt.axis('off'),plt.title("6. mask add"),plt.imshow(cv.cvtColor(imgAddCir, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def ImageWeighted(filepath1, filepath2):
    img1 = cv.imread(filepath1)
    img2 = cv.imread(filepath2)

    h,w = img1.shape[:2]
    img3 = cv.resize(img2, (w,h))
    print(img1.shape, img2.shape, img3.shape)

    # 两幅图像的加权加法，推荐 alpha + beta = 1.0
    alpha ,beta = 0.25, 0.75
    imgAddW1 = cv.addWeighted(img1, alpha, img3, beta, 0)

    alpha ,beta = 0.5, 0.5
    imgAddW2 = cv.addWeighted(img1, alpha, img3, beta, 0)

    alpha ,beta = 0.75, 0.25
    imgAddW3 = cv.addWeighted(img1, alpha, img3, beta, 0)

    # 两幅图的渐进变化
    wList = np.arange(0.1, 1.0, 0.05)
    print(wList)
    for weight in wList:
        imgWeight = cv.addWeighted(img1, weight, img3, (1-weight), 0)
        cv.imshow("ImageAddWeight", imgWeight)
        cv.waitKey(100)
    cv.destroyAllWindows()

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. a=0.2, b=0.8"),plt.imshow(cv.cvtColor(imgAddW1, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. a=0.5  b=0.5"),plt.imshow(cv.cvtColor(imgAddW2, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. a=0.8  b=0.8"),plt.imshow(cv.cvtColor(imgAddW3, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def ImageMeanFiltering(filepath1):
    img = cv.imread(filepath1, flags= 0)  # 灰度图像
    H,W = img.shape[:2]

    k = 15 # 均值滤波器的尺寸

    # 两重循环卷积运算实现均值滤波
    timeBegin = cv.getTickCount()
    pad = k//2 +1
    imgPad = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT)
    imgBox1 = np.zeros((H,W), np.int32)
    for h in range(H):
        for w in range(W):
            imgBox1[h,w] = np.sum(imgPad[h:h+k, w:w+k]) / (k*k)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Blurred by double cycle : {} sec".format(round(time,4)))

    # 基于积分图像方法实现均值滤波
    timeBegin = cv.getTickCount()
    pad = k//2 +1
    imgPadded = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT)
    sumImg = cv.integral(imgPadded)
    imgBox2 = np.zeros((H,W), np.uint8)
    imgBox2[:,:] = (sumImg[:H,:W] - sumImg[:H, k:W+k] - sumImg[k:H+k,:W] + sumImg[k:H+k, k:W+k])/ (k*k)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Blurred by intergral image : {} sec".format(round(time,4)))

    # 基于cv.boxfilter均值滤波
    timeBegin = cv.getTickCount()
    kernel = np.zeros(k, np.float32)/(k*k)
    imgBoxF = cv.boxFilter(img,-1,(k,k))
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Blurred by cv.boxFilter : {} sec".format(round(time,4)))

    plt.figure(figsize=(9,6))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. blurred by dual cycle"),plt.imshow(imgBox1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. blurred by intergral image"),plt.imshow(imgBox2, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__" :
    filepath1 = "/home/calmcar/work/python/opencv_test/img/lenna.bmp"
    filepath2 = "/home/calmcar/work/python/opencv_test/img/monarch.bmp"

    # ImageAdd(filepath1, filepath2)
    # ImageMask(filepath1, filepath2)
    # ImageWeighted(filepath1, filepath2)
    ImageMeanFiltering(filepath1)
