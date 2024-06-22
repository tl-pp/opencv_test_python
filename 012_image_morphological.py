import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 形态学图像处理

def ImageErodeAndDilate(filepath):
    img = cv.imread(filepath, flags=0)

    # 图像反转
    transtable = np.array([(255 - i) for i in range(256)]).astype(np.uint8)
    invLut = cv.LUT(img, transtable)

    _, imgBin = cv.threshold(invLut, 20, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理

    # 图像腐蚀
    ksize1 = (3, 3)  # 结构元尺寸3x3
    kernel1 = np.ones(ksize1, dtype=np.uint8)     # 矩形结构元
    imgErode1 = cv.erode(imgBin, kernel=kernel1)  # 图像腐蚀
    kernel2 = np.ones((9, 9), dtype=np.uint8)
    imgErode2 = cv.erode(imgBin, kernel=kernel2)
    imgErode3 = cv.erode(imgBin, kernel=kernel1, iterations=2)  # 腐蚀两次

    # 图像膨胀
    ksize1 = (3, 3)  # 结构元尺寸3x3
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, ksize1)  # 矩形结构元
    imgDilatel = cv.dilate(imgBin, kernel=kernel1) # 图像膨胀
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    imgDilate2 = cv.dilate(imgBin, kernel=kernel2)
    imgDilate3 = cv.dilate(imgBin, kernel=kernel1, iterations=2)  # 膨胀两次

    # 对腐蚀的图像膨胀
    dilateErode = cv.dilate(imgErode1, kernel=kernel1)

    plt.figure(figsize=(9,5))
    plt.subplot(241),plt.axis('off'),plt.title("1. Original"),plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(242),plt.axis('off'),plt.title("2. Erode size=(3, 3)"),plt.imshow(imgErode1, cmap='gray')
    plt.subplot(243),plt.axis('off'),plt.title("3. Erode size=(9, 9)"),plt.imshow(imgErode2, cmap='gray')
    plt.subplot(244),plt.axis('off'),plt.title("4. Erode size=(3, 3) * 2"),plt.imshow(imgErode3, cmap='gray')
    plt.subplot(245),plt.axis('off'),plt.title("5. Erode & Dilated"),plt.imshow(dilateErode, cmap='gray')
    plt.subplot(246),plt.axis('off'),plt.title("6. Dilate size=(3, 3)"),plt.imshow(imgDilatel, cmap='gray')
    plt.subplot(247),plt.axis('off'),plt.title("7. Dilate size=(9, 9)"),plt.imshow(imgDilate2, cmap='gray')
    plt.subplot(248),plt.axis('off'),plt.title("8. Dilate size=(3, 3) * 2"),plt.imshow(imgDilate3, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageOpenAndClose(filepath):
    img = cv.imread(filepath, flags=0)
    # 图像反转
    transtable = np.array([(255 - i) for i in range(256)]).astype(np.uint8)
    invLut = cv.LUT(img, transtable)

    _, imgBin = cv.threshold(invLut, 20, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理

    # 图像腐蚀
    ksize = (5, 5)  # 结构元尺寸
    elemet =  cv.getStructuringElement(cv.MORPH_RECT, ksize)     # 矩形结构元
    imgErode = cv.erode(imgBin, kernel=elemet)  # 图像腐蚀
    # 对腐蚀图像进行膨胀
    imgDilateErode = cv.dilate(imgErode, kernel=elemet)  # 腐蚀->膨胀

    # 图像的开运算
    imgOpen = cv.morphologyEx(imgBin, cv.MORPH_OPEN, kernel=elemet)

    # 图像膨胀
    ksize = (5, 5)  # 结构元尺寸
    elemet =  cv.getStructuringElement(cv.MORPH_RECT, ksize)     # 矩形结构元
    imgDilate = cv.dilate(imgBin, kernel=elemet)  # 图像膨胀
    # 对膨胀图像进行腐蚀
    imgErodeDilate = cv.erode(imgDilate, kernel=elemet)

    # 图像的闭运算
    imgClose = cv.morphologyEx(imgBin, cv.MORPH_CLOSE, kernel=elemet)

    plt.figure(figsize=(9,5))
    plt.subplot(241),plt.axis('off'),plt.title("1. Original"),plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(242),plt.axis('off'),plt.title("2. Eroded"),plt.imshow(imgErode, cmap='gray')
    plt.subplot(243),plt.axis('off'),plt.title("3. Eroded & Dilated"),plt.imshow(imgDilateErode, cmap='gray')
    plt.subplot(244),plt.axis('off'),plt.title("4. Opening"),plt.imshow(imgOpen, cmap='gray')
    plt.subplot(245),plt.axis('off'),plt.title("5. Binary"),plt.imshow(imgBin, cmap='gray')
    plt.subplot(246),plt.axis('off'),plt.title("6. Dilated"),plt.imshow(imgDilate, cmap='gray')
    plt.subplot(247),plt.axis('off'),plt.title("7. Dilated & Eroded"),plt.imshow(imgErodeDilate, cmap='gray')
    plt.subplot(248),plt.axis('off'),plt.title("8. Closing"),plt.imshow(imgClose, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageGradient(filepath):
    img = cv.imread(filepath, flags=0)
    # 图像反转
    transtable = np.array([(255 - i) for i in range(256)]).astype(np.uint8)
    invLut = cv.LUT(img, transtable)

    _, imgBin = cv.threshold(invLut, 20, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理

    # 图像的形态学梯度运算
    element = cv.getStructuringElement(cv.MORPH_RECT, (3,3))     # 矩形结构元
    imgGrad = cv.morphologyEx(imgBin, cv.MORPH_GRADIENT, kernel=element)

    # 开运算->形态学梯度运算
    imgOpen = cv.morphologyEx(imgBin, cv.MORPH_OPEN, kernel=element)
    imgOpenGrand = cv.morphologyEx(imgOpen, cv.MORPH_GRADIENT, kernel=element)

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Gradient"),plt.imshow(imgGrad, cmap='gray')
    plt.subplot(133),plt.axis('off'),plt.title("3. Opening -> Gradient"),plt.imshow(imgOpenGrand, cmap='gray')
    plt.tight_layout()
    plt.show()

def ImageHitMiss(filepath):
    img = cv.imread(filepath, flags=0)

    # 图像反转
    transtable = np.array([(255 - i) for i in range(256)]).astype(np.uint8)
    invLut = cv.LUT(img, transtable)

    _, binary = cv.threshold(invLut, 20, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理

    kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))   # 圆形结构元
    imgBin = cv.morphologyEx(binary, cv.MORPH_CLOSE, kern)     # 封闭孔洞

    # 击中-击不中变换
    kernB1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12,12))
    imgHMT1 = cv.morphologyEx(imgBin, cv.MORPH_HITMISS, kernB1)
    kernB2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))
    imgHMT2 = cv.morphologyEx(imgBin, cv.MORPH_HITMISS, kernB2)

    
    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. HITMISS (12,12)"),plt.imshow(imgHMT1, cmap='gray')
    plt.subplot(133),plt.axis('off'),plt.title("3. HITMISS (20,20)"),plt.imshow(imgHMT2, cmap='gray')
    plt.tight_layout()
    plt.show()

def ImageGrayScale(filepath):
    img = cv.imread(filepath, flags=0)

    element = cv.getStructuringElement(cv.MORPH_RECT, (3,3))     # 矩形结构元
    imgErode = cv.erode(img, element)                            # 灰度腐蚀
    imgDilate = cv.dilate(img, element)                          # 灰度膨胀
    imgOpen = cv.morphologyEx(img, cv.MORPH_OPEN, element)       # 灰度开运算
    imgClose = cv.morphologyEx(img, cv.MORPH_CLOSE, element)     # 灰度闭运算
    imgGrad = cv.morphologyEx(img, cv.MORPH_GRADIENT, element)   # 灰度梯度运算

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232),plt.axis('off'),plt.title("2. GrayScal Erode"),plt.imshow(imgErode, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. GrayScal Dilate"),plt.imshow(imgDilate, cmap='gray')
    plt.subplot(234),plt.axis('off'),plt.title("4. GrayScal Open"),plt.imshow(imgOpen, cmap='gray')
    plt.subplot(235),plt.axis('off'),plt.title("5. GrayScal Close"),plt.imshow(imgClose, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. GrayScal Gradient"),plt.imshow(imgGrad, cmap='gray')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filepath1 = r"img/Fig1201.png"
    filepath2 = r"img/Fig0703.png"
    filepath3 = r"img/Fig1202.png"
    filepath4 = r"img/Fig1101.png"

    # 形态学之腐蚀与膨胀
    # ImageErodeAndDilate(filepath1)

    # 形态学之开运算与闭运算
    # ImageOpenAndClose(filepath1)

    # 形态学之梯度运算
    # ImageGradient(filepath2)

    # 用击中-击不中变换进行特征识别
    # ImageHitMiss(filepath3)

    # 灰度形态学运算
    ImageGrayScale(filepath4)

