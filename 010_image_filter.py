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


def ImageMedianBlur(filepath):
    img = cv.imread(filepath, flags=0)

    # (1) 高斯低通滤波器
    ksize = (11, 11)
    GaussBlur = cv.GaussianBlur(img, ksize, 0)

    # (2) 中值滤波器
    medianBlur = cv.medianBlur(img, ksize=5)

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. GaussianFilter(k=11)"),plt.imshow(GaussBlur, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. MedianBlur(size=3)"),plt.imshow(medianBlur, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageOrderingFilter(filepath):
    img = cv.imread(filepath, flags=0)
    hImg, wImg = img.shape[:2]

    # 边界填充
    m, n = 3, 3  # 统计排序滤波器尺寸
    hPad, wPad = int((m - 1)/2), int((n - 1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    imgMedianF = np.zeros(img.shape)     # 中值滤波器
    imgMaximumF = np.zeros(img.shape)    # 最大值滤波器
    imgMinimumF = np.zeros(img.shape)    # 最小值滤波器
    imgMiddleF = np.zeros(img.shape)     # 中点滤波器
    imgAlphaF = np.zeros(img.shape)      # 修正阿尔法均值滤波器

    for h in range(hImg):
        for w in range(wImg):
            # 当前像素的临域
            neighborhood = imgPad[h:h+m, w:w+n]
            padMax = np.max(neighborhood)   # 临值域最大
            padMin = np.min(neighborhood)   # 临值域最小

            # (1) 中值滤波器
            imgMedianF[h,w] = np.median(neighborhood)

            # (2) 最大值滤波器
            imgMaximumF[h,w] = padMax

            # (3) 最小值滤波器
            imgMinimumF[h,w] = padMin

            # (4) 中点滤波器
            imgMiddleF[h,w] = int(padMax/2 + padMin/2)

            # (5) 修正Alpha均值滤波器
            d = 2    # 修正值
            neighborSort = np.sort(neighborhood.flatten())   # 邻域像素按灰度值排序
            sumAlpha = np.sum(neighborSort[d:m*n-d-1])       # 删除d个最大灰度值，d个最小灰度值
            imgAlphaF[h,w] = sumAlpha / (m*n -2*d)           # 对剩余像素进行算术平均

    plt.figure(figsize=(9,6.5))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232),plt.axis('off'),plt.title("2. Median  filter"),plt.imshow(imgMedianF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233),plt.axis('off'),plt.title("3. Maximum filter"),plt.imshow(imgMaximumF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234),plt.axis('off'),plt.title("4. Minumun filter"),plt.imshow(imgMinimumF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(235),plt.axis('off'),plt.title("5. Middle  filter"),plt.imshow(imgMiddleF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236),plt.axis('off'),plt.title("6. Alpha   filter"),plt.imshow(imgAlphaF, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
    
            
def ImageAdaptationBoxFilter(filepath):
    img = cv.imread(filepath, flags=0)
    hImg, wImg = img.shape[:2]

    # 边界填充
    m, n = 5, 5  # 统计排序滤波器尺寸 mxn矩形邻域
    hPad, wPad = int((m - 1)/2), int((n - 1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    # 估计原始图像的噪声方差 VarImg
    mean, stddev = cv.meanStdDev(img)    # 图像均值、方差
    varImg = stddev ** 2

    # 自适应局部降噪
    epsilon = 1e-8
    imgAdaptLocal = np.zeros(img.shape)

    for h in range(hImg):
        for w in range(wImg):
            # 当前像素的临域
            neighborhood = imgPad[h:h+m, w:w+n]
            meanSxy, stddevSxy = cv.meanStdDev(neighborhood)   # 邻域局部均值
            varSxy = stddevSxy ** 2
            ratioVar = min(varImg / (varSxy + epsilon), 1.0)   # 加性噪声 varImg < varSxy
            imgAdaptLocal[h,w] = img[h,w] - ratioVar * (img[h,w] - meanSxy)

    # 均值滤波器， 用于比较
    imgAriMean = cv.boxFilter(img, -1, (m,n))

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Box filter"),plt.imshow(imgAriMean, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Adapt local filter"),plt.imshow(imgAdaptLocal, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()






    


if __name__ == '__main__':
    filepath1 = r"img/Fig1001.png"
    filepath2 = r"img/Fig1002.png"
    

    # 图像的卷积运算与相关运算
    # ImageConvolution(filepath1)

    # 空间滤波之盒式滤波器
    # ImageBoxFilter(filepath1)

    # 空间滤波之高斯滤波器
    # ImageGaussianBlur(filepath1)

    # 空间滤波之中值滤波器
    # ImageMedianBlur(filepath2)

    # 空间滤波器之排序滤波器
    # ImageOrderingFilter(filepath2)

    # 空间滤波器之自适应滤波器
    # ImageAdaptationBoxFilter(filepath2)
