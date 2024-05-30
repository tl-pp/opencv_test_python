import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 图像的灰度变换

def ImageGrayscaleFlip(filepath1):
    img = cv.imread(filepath1, flags = 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # LUT快速查表法
    transTable = np.array([(255 - i) for i in range(256)]).astype("uint8")
    imgInv = cv.LUT(img, transTable)   # 彩色图像灰度翻转变换
    grayInv = cv.LUT(gray, transTable) # 灰度图像灰度翻转变换

    print(img.shape, imgInv.shape, grayInv.shape)

    plt.figure(figsize=(7,5))
    plt.subplot(221),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(222),plt.axis('off'),plt.title("2. gray"),plt.imshow(gray, cmap='gray')
    plt.subplot(223),plt.axis('off'),plt.title("3. Invert img"),plt.imshow(cv.cvtColor(imgInv, cv.COLOR_BGR2RGB))
    plt.subplot(224),plt.axis('off'),plt.title("4. Invert gray"),plt.imshow(grayInv, cmap='gray')
    
    plt.tight_layout()
    plt.show()

def ImageGrayLinearMapping(filepath1):
    gray = cv.imread(filepath1, flags = 0)
    h, w = gray.shape[:2]

    # 线性变换 dst = a * src + b
    a1, b1 = 1, 50    # a = 1, b > 0 : 灰度值向上偏移
    a2, b2 = 1, -50   # a = 1, b < 0 : 灰度值向下偏移
    a3, b3 = 1.5, 0   # a > 1, b = 0 : 对比度增强
    a4, b4 = 0.8, 0   # a < 1, b = 0 : 对比度减弱
    a5, b5 = -0.5, 0  # a < 0, b = 0 : 暗区域变亮，亮区域变暗
    a6, b6 = -1, 255  # a = -1, b = 255 : 灰度值翻转

    # 灰度线性变换
    timeBegin = cv.getTickCount()
    img1 = cv.convertScaleAbs(gray, alpha=a1, beta=b1)
    img2 = cv.convertScaleAbs(gray, alpha=a2, beta=b2)
    img3 = cv.convertScaleAbs(gray, alpha=a3, beta=b3)
    img4 = cv.convertScaleAbs(gray, alpha=a4, beta=b4)
    img5 = cv.convertScaleAbs(gray, alpha=a5, beta=b5)
    img6 = cv.convertScaleAbs(gray, alpha=a6, beta=b6)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Grayscale transformtion by opencv:{} sec".format(round(time,4)))

    # 二重循环遍历
    timeBegin = cv.getTickCount()
    for i in range(h):
        for j in range(w):
            img1[i][j] = min(255, max((gray[i][j] + b1), 0))
            img2[i][j] = min(255, max((gray[i][j] + b2), 0))
            img3[i][j] = min(255, max((a3 * gray[i][j]), 0))
            img4[i][j] = min(255, max((a4 * gray[i][j]), 0))
            img5[i][j] = min(255, max(abs(a5 * gray[i][j] + b5), 0))
            img6[i][j] = min(255, max(abs(a6 * gray[i][j] + b6), 0))
    
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Grayscale transformtion by nested loop:{} sec".format(round(time,4)))

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. a = 1, b = 50"),plt.imshow(img1, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. a = 1, b = -50"),plt.imshow(img2, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. a = 1.5, b = 0"),plt.imshow(img3, cmap='gray')
    plt.subplot(234),plt.axis('off'),plt.title("4. a = 0.8, b = 0"),plt.imshow(img4, cmap='gray')
    plt.subplot(235),plt.axis('off'),plt.title("5. a = -0.5, b = 0"),plt.imshow(img5, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. a = -1, b = 255"),plt.imshow(img6, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageHistogramNormalization(filepath1):
    gray = cv.imread(filepath1, flags = 0)
    h, w = gray.shape[:2]

    # 直方图正规化
    gray = cv.add(cv.multiply(gray, 0.6), 36)  # 调整灰度范围

    iMax, iMin = np.max(gray), np.min(gray)
    oMax, oMin = 255, 0
    a = float((oMax - oMin) / (iMax - iMin))
    b = oMin - a * iMin
    dst = a * gray + b
    grayNorm1 = dst.astype(np.uint8)

    grayNorm2 = cv.normalize(gray, None, 0 , 255, cv.NORM_MINMAX)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. gray"),plt.imshow(gray, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. normalized"),plt.imshow(grayNorm1, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. cv.normalize"),plt.imshow(grayNorm2, cmap='gray')

    plt.subplot(234),plt.axis('off'),plt.title("4. gray histogram")
    histCV = cv.calcHist([gray], [0], None, [256], [0, 255])
    plt.bar(range(256), histCV[:, 0])
    plt.axis([0, 255, 0, np.max(histCV)])

    plt.subplot(235),plt.axis('off'),plt.title("5. normalized histogram")
    histCV1 = cv.calcHist([grayNorm1], [0], None, [256], [0, 255])
    plt.bar(range(256), histCV1[:, 0])
    plt.axis([0, 255, 0, np.max(histCV)])

    plt.subplot(236),plt.axis('off'),plt.title("6. cv.normalize histogram")
    histCV2 = cv.calcHist([grayNorm2], [0], None, [256], [0, 255])
    plt.bar(range(256), histCV2[:, 0])
    plt.axis([0, 255, 0, np.max(histCV)])

    plt.tight_layout()
    plt.show()




if __name__ == "__main__" :
    filepath1 = r"img/lenna.bmp"
    filepath2 = r"img/remap.png"

    # 图像翻转变换
    # ImageGrayscaleFlip(filepath1)

    # 图像线性变换
    # ImageGrayLinearMapping(filepath1)

    # 图像直方图归一化
    ImageHistogramNormalization(filepath1)
