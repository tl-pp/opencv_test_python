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








if __name__ == '__main__':
    filepath1 = r"img/Lena.tif"
    filepath2 = r"img/Fig0301.png"
    

    # 图像的卷积运算与相关运算
    ImageConvolution(filepath1)