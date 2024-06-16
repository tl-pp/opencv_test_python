import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 图像卷积与空间滤波

def ImageConvolution(filepath):
    img = cv.imread(filepath, flags=0)

    # (1)不对称卷积核
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])   # 不对称卷积核
    imgCorr = cv.filter2D(img, -1, kernel)  # 相关运算
    kernFlip = cv.flip(kernel, -1)          # 翻转卷积核
    imgConv = cv.filter2D(img, -1, kernFlip)

    print("(1) Asymmetric convolution kernel")
    print("\tCompare imgCorr & imgConv: ", (imgCorr == imgConv).all())
    print("kernel:\n{} \nkernFlip:\n{}".format(kernel, kernFlip))

    # (2)对称卷积核
    kernSymm = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # 对称卷积核
    imgCorrSym = cv.filter2D(img, -1, kernSymm)
    kernFlip = cv.flip(kernSymm, -1)        # 卷积核旋转180°
    imgConvSym = cv.filter2D(img, -1, kernFlip)

    print("(2) Symmetric convolution kernel")
    print("\tCompare imgCorr & imgConv: ", (imgCorrSym == imgConvSym).all())
    print("kernSymm:\n{} \nkernFlip:\n{}".format(kernSymm, kernFlip))

    # (3)可分离卷积核 kernXY = kernX * kernY
    kernX = np.array([[-1, 2, -1]], np.float32)  # 水平卷积核(1, 3)
    kernY = np.transpose(kernX)                  # 垂直卷积核(3, 1)
    kernXY = kernX * kernY                       # 二维卷积核(3, 3)
    kFilp = cv.flip(kernXY, -1)                  # 水平和垂直翻转卷积核
    imgConvXY = cv.filter2D(img, -1, kernXY)
    imgConvSep = cv.sepFilter2D(img, -1, kernX, kernY)

    print("(3) Separable convolution kernel")
    print("\tCompare imgConvXY & imgConvSep: ", (imgConvXY == imgConvSep).all())
    # print("kernX:{}, kernY:{}, kernXY:{}, kFilp:{}".format(kernX.shape, kernY.shape, kernXY.shape, kFilp.shape))
    print("kernX:\n{} \nkernY:\n{} \nkernXY:\n{} \nkFilp:\n{}".format(kernX, kernY, kernXY, kFilp))

    plt.figure(figsize=(9,3))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Correlation"),plt.imshow(imgCorr, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Convolution"),plt.imshow(imgConv, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageBoxFilter(filepath):
    img = cv.imread(filepath, flags=0)

    # (1) 盒式滤波器 3种实现方式
    ksize = (5, 5)
    kernel = np.ones(ksize, np.float32) / (ksize[0] * ksize[1])   # 生成归一化核

    conv1 = cv.filter2D(img, -1, kernel)
    conv2 = cv.blur(img, ksize)
    conv3 = cv.boxFilter(img, -1, ksize)

    print("Compare conv1 & conv2: ", (conv1 == conv2).all())
    print("Compare conv1 & conv3: ", (conv1 == conv3).all())

    # (2) 滤波器尺寸的影响
    imgConv1 = cv.blur(img, (5, 5))
    imgConv2 = cv.blur(img, (11, 11))

    plt.figure(figsize=(9,3.2))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. boxFilter(5, 5)"),plt.imshow(imgConv1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. boxFilter(11, 11)"),plt.imshow(imgConv2, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

def ImageGaussianBlur(filepath):
    img = cv.imread(filepath, flags=0)

    # (1)计算高斯核
    kernX = cv.getGaussianKernel(5, 0)  # 一维高斯和
    kernel = kernX * kernX.T  # 二维高斯核
    print("1D kernel of Gaussian:{}".format(kernX.shape))
    print(kernX.round(4))
    print(kernX.T.round(4))
    print("2D kernel of Gaussian:{}".format(kernel.shape))
    print(kernel.round(4))

    # (2)高斯低通滤波器
    ksize = (11, 11)   # 高斯滤波器核的尺寸
    GaussBlur11 = cv.GaussianBlur(img, ksize, 0)   # sigma由ksize计算
    ksize = (43, 43)
    GaussBlur43 = cv.GaussianBlur(img, ksize, 0)

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. GaussianFilter(k=11)"),plt.imshow(GaussBlur11, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. GaussianFilter(k=43)"),plt.imshow(GaussBlur43, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filepath1 = r"img/Fig1001.png"
    filepath2 = r"img/Fig0301.png"
    

    # 图像的卷积运算与相关运算
    # ImageConvolution(filepath1)

    # 空间滤波之盒式滤波器
    # ImageBoxFilter(filepath1)

    # 空间滤波之高斯滤波器
    ImageGaussianBlur(filepath1)
