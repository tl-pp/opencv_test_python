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
    m, n = 5, 5  # 滤波器尺寸 mxn矩形邻域
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


def ImageAdaptationMedFilter(filepath):
    img = cv.imread(filepath, flags=0)
    hImg, wImg = img.shape[:2]
    print(hImg, wImg)

    # 边界填充
    smax = 7
    m, n = smax, smax    # 滤波器尺寸 mxn矩形邻域
    hPad, wPad = int((m - 1)/2), int((n - 1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    imgMedianFilter = np.zeros(img.shape)
    imgAdaptMedFilter = np.zeros(img.shape)

    for h in range(hPad, hPad + hImg):
        for w in range(wPad, wPad+ wImg):
            # (1)中值滤波器
            ksize = 3
            kk = ksize // 2
            win = imgPad[h-kk:h+kk+1, w-kk:w+kk+1]
            imgMedianFilter[h-hPad, w-wPad] = np.median(win)

            # (2)自适应中值滤波器
            ksize = 3
            zxy = img[h-hPad, w-wPad]
            while True:
                k = ksize//2
                win = imgPad[h-k:h+k+1, w-k:w+k+1]
                zmin, zmed, zmax = np.min(win), np.median(win), np.max(win)

                if zmin < zmed < zmax :   # zmed 不是噪声
                    if zmin < zxy < zmax:
                        imgAdaptMedFilter[h-hPad, w-wPad] = zxy
                    else:
                        imgAdaptMedFilter[h-hPad, w-wPad] = zmed
                    break
                else:
                    if ksize >= smax:  # 达到最大窗口
                        imgAdaptMedFilter[h-hPad, w-wPad] = zmed
                        break
                    else:              # 未达到最大窗口
                        ksize = ksize+2    # 增大窗口尺寸
                    
    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Median filter"),plt.imshow(imgMedianFilter, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Adapt Median filter"),plt.imshow(imgMedianFilter, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageBilateralFilter(filepath):
    img = cv.imread(filepath, flags=0)
    hImg, wImg = img.shape[:2]
    print(hImg, wImg)

    # (1) 高斯滤波核
    ksize = (11, 11)
    ImageGaussianF = cv.GaussianBlur(img, ksize, 0, 0)
    
    # (2) 双边滤波器
    imgBilateralF = cv.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=10)

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Gaussian filter"),plt.imshow(ImageGaussianF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Bilateral filter"),plt.imshow(imgBilateralF, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImagePassivation(filepath):
    img = cv.imread(filepath, flags=0)
    hImg, wImg = img.shape[:2]
    print(hImg, wImg)

    # (1) 对原始图像进行高斯平滑
    imgGauss = cv.GaussianBlur(img, (11, 11), sigmaX=5.0)

    # (2) 掩蔽模版： 从原始图像中减去平滑图像
    mashPassivate = cv.subtract(img, imgGauss)

    # (3) 掩蔽模版与原始图像相加
    # k<1 减弱钝化掩蔽
    maskWeak = cv.multiply(mashPassivate, 0.5)
    passivation1 = cv.add(img,maskWeak)

    # k=1 钝化掩蔽
    passivation2 = cv.add(img, mashPassivate)

    # k>1 高提升滤波
    maskEnhance = cv.multiply(mashPassivate,2.0)
    passivation3 = cv.add(img, maskEnhance)

    plt.figure(figsize=(9,6.5))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232),plt.axis('off'),plt.title("2. GaussBlur"),plt.imshow(imgGauss, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233),plt.axis('off'),plt.title("3. PassivateMask"),plt.imshow(mashPassivate, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234),plt.axis('off'),plt.title("4. passivation(k=0.5)"),plt.imshow(passivation1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(235),plt.axis('off'),plt.title("5. passivation(k=1.0)"),plt.imshow(passivation2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236),plt.axis('off'),plt.title("6. passivation(k=2.0)"),plt.imshow(passivation3, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageLaplacian(filepath):
    img = cv.imread(filepath, flags=0)

    # (1)使用函数 cv.filter2D 计算laplacian K1 K2
    LaplacianK1 = np.array([[0,1,0], [1,-4,1], [0,1,0]])    # K1
    imgLapK1 = cv.filter2D(img, -1, LaplacianK1, cv.BORDER_REFLECT)
    LaplacianK2 = np.array([[1,1,1], [1,-8,1],[1,1,1]])
    imgLapK2 = cv.filter2D(img, -1, LaplacianK2, cv.BORDER_REFLECT)

    # (2)使用函数 cv.laplacian 计算 Laplacian
    imgLaplacian = cv.Laplacian(img, cv.CV_32F, ksize=3)   # 输出为浮点类型数据
    abslaplacian = cv.convertScaleAbs(imgLaplacian)        # 拉伸到[0,255]
    print(type(imgLaplacian[0,0]), type(abslaplacian[0,0]))
    print(imgLaplacian)
    print(abslaplacian)

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Laplacian(float)"),plt.imshow(imgLaplacian, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Laplacian(abs)"),plt.imshow(abslaplacian, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageSobel(filepath):
    img = cv.imread(filepath, flags=0)

    # (1) 使用函数cv.filter2D 实现
    kernSobelX = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    kernSobelY = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    SobelX = cv.filter2D(img, -1, kernSobelX)
    SobelY = cv.filter2D(img, -1, kernSobelY)

    # (2) 使用函数cv.Sobel 实现
    imgSobelX = cv.Sobel(img, cv.CV_16S, 1, 0)    # x轴方向
    imgSobelY = cv.Sobel(img, cv.CV_16S, 0, 1)    # x轴方向
    absSobelX = cv.convertScaleAbs(imgSobelX)
    absSobelY = cv.convertScaleAbs(imgSobelY)
    SobelXY = cv.add(absSobelX, absSobelY)

    plt.figure(figsize=(9,6.5))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232),plt.axis('off'),plt.title("2. SobelX(float)"),plt.imshow(imgSobelX, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233),plt.axis('off'),plt.title("3. SobelYfloat)"),plt.imshow(imgSobelY, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234),plt.axis('off'),plt.title("4. SobelXY(abs)"),plt.imshow(SobelXY, cmap='gray', vmin=0, vmax=255)
    plt.subplot(235),plt.axis('off'),plt.title("5. SobelX(abs)"),plt.imshow(absSobelX, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236),plt.axis('off'),plt.title("6. SobelY(abs)"),plt.imshow(absSobelY, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageScharr(filepath):
    img = cv.imread(filepath, flags=0)

    # (1) 使用函数cv.filter2D 实现
    kernScharrX = np.array([[-3,0,3], [-10,0,10], [-3,0,3]])
    kernScharrY = np.array([[-3,10,-3], [0,0,0], [3,10,3]])
    ScharrX = cv.filter2D(img, -1, kernScharrX)
    ScharrY = cv.filter2D(img, -1, kernScharrY)

    # (2) 使用函数cv.Scharr 实现
    imgScharrX = cv.Scharr(img, cv.CV_16S, 1, 0)    # x轴方向
    imgScharrY = cv.Scharr(img, cv.CV_16S, 0, 1)    # x轴方向
    absScharrX = cv.convertScaleAbs(imgScharrX)
    absScharrY = cv.convertScaleAbs(imgScharrY)
    ScharrXY = cv.add(absScharrX, absScharrY)

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. ScharrX(abs)"),plt.imshow(absScharrX, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. ScharrY(abs)"),plt.imshow(absScharrY, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. ScharrXY(abs)"),plt.imshow(ScharrXY, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageGaussianPyramid(filepath):
    img = cv.imread(filepath, flags=1)

    # 图像向下采样
    pyrG0 = img.copy()        # G0 (512,512)
    pyrG1 = cv.pyrDown(pyrG0)    # G1 (256,256)
    pyrG2 = cv.pyrDown(pyrG1)    # G2 (128,128)
    pyrG3 = cv.pyrDown(pyrG2)    # G3 (64,64)
    print(pyrG0.shape, pyrG1.shape, pyrG2.shape, pyrG3.shape)
    
    # 图像向上采样
    pyrU3 = pyrG3.copy()
    pyrU2 = cv.pyrUp(pyrU3)
    pyrU1 = cv.pyrUp(pyrU2)
    pyrU0 = cv.pyrUp(pyrU1)
    print(pyrU0.shape, pyrU1.shape, pyrU2.shape, pyrU3.shape)

    plt.figure(figsize=(9,5))
    plt.subplot(241),plt.axis('off'),plt.title("G0 " + str(pyrG0.shape[:2])),plt.imshow(cv.cvtColor(pyrG0, cv.COLOR_BGR2RGB))
    plt.subplot(242),plt.axis('off'),plt.title("->G1 " + str(pyrG1.shape[:2]))
    down1 = np.ones_like(img, dtype=np.uint8) * 128
    down1[:pyrG1.shape[0], :pyrG1.shape[1], :] = pyrG1
    plt.imshow(cv.cvtColor(down1, cv.COLOR_BGR2RGB))
    
    plt.subplot(243),plt.axis('off'),plt.title("->G2 " + str(pyrG2.shape[:2]))
    down2 = np.ones_like(img, dtype=np.uint8) * 128
    down2[:pyrG2.shape[0], :pyrG2.shape[1], :] = pyrG2
    plt.imshow(cv.cvtColor(down2, cv.COLOR_BGR2RGB))

    plt.subplot(244),plt.axis('off'),plt.title("->G3 " + str(pyrG3.shape[:2]))
    down3 = np.ones_like(img, dtype=np.uint8) * 128
    down3[:pyrG3.shape[0], :pyrG3.shape[1], :] = pyrG3
    plt.imshow(cv.cvtColor(down3, cv.COLOR_BGR2RGB))
    
    plt.subplot(245),plt.axis('off'),plt.title("U0 " + str(pyrU0.shape[:2]))
    up0 = np.ones_like(img, dtype=np.uint8) * 128
    up0[:pyrU0.shape[0], :pyrU0.shape[1], :] = pyrU0
    plt.imshow(cv.cvtColor(up0, cv.COLOR_BGR2RGB))

    plt.subplot(246),plt.axis('off'),plt.title("U1 " + str(pyrU1.shape[:2]))
    up1 = np.ones_like(img, dtype=np.uint8) * 128
    up1[:pyrU1.shape[0], :pyrU1.shape[1], :] = pyrU1
    plt.imshow(cv.cvtColor(up1, cv.COLOR_BGR2RGB))

    plt.subplot(247),plt.axis('off'),plt.title("U2 " + str(pyrU2.shape[:2]))
    up2 = np.ones_like(img, dtype=np.uint8) * 128
    up2[:pyrU2.shape[0], :pyrU2.shape[1], :] = pyrU2
    plt.imshow(cv.cvtColor(up2, cv.COLOR_BGR2RGB))

    plt.subplot(248),plt.axis('off'),plt.title("U3 " + str(pyrU3.shape[:2]))
    up3 = np.ones_like(img, dtype=np.uint8) * 128
    up3[:pyrU3.shape[0], :pyrU3.shape[1], :] = pyrU3
    plt.imshow(cv.cvtColor(up3, cv.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.show()


def ImageLaplacianPyramid(filepath):
    img = cv.imread(filepath, flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 图像向下取样， 构造高斯金字塔
    pyrG0 = img.copy()           # G0 (512,512)
    pyrG1 = cv.pyrDown(pyrG0)    # G1 (256,256)
    pyrG2 = cv.pyrDown(pyrG1)    # G2 (128,128)
    pyrG3 = cv.pyrDown(pyrG2)    # G3 (64,64)
    pyrG4 = cv.pyrDown(pyrG3)    # G3 (32,32)
    print(pyrG0.shape, pyrG1.shape, pyrG2.shape, pyrG3.shape, pyrG4.shape)

    # 构造拉普拉斯金字塔， 高斯金字塔的每层减去其上层图像的上采样
    pyrL0 = pyrG0 - cv.pyrUp(pyrG1)    # L0 (512,512)
    pyrL1 = pyrG1 - cv.pyrUp(pyrG2)    # L1 (256,256)
    pyrL2 = pyrG2 - cv.pyrUp(pyrG3)    # L2 (128,128)
    pyrL3 = pyrG3 - cv.pyrUp(pyrG4)    # L3 (64,64)
    print(pyrL0.shape, pyrL1.shape, pyrL2.shape, pyrL3.shape)

    # 向上采样恢复高分辨率
    rebuildG3 = pyrL3 + cv.pyrUp(pyrG4)
    rebuildG2 = pyrL2 + cv.pyrUp(rebuildG3)
    rebuildG1 = pyrL1 + cv.pyrUp(rebuildG2)
    rebuildG0 = pyrL0 + cv.pyrUp(rebuildG1)
    print(rebuildG0.shape, rebuildG1.shape, rebuildG2.shape, rebuildG3.shape)

    print("diff of rebuild: ", np.mean(abs(rebuildG0 - img)))

    plt.figure(figsize=(10,8))
    plt.subplot(341),plt.axis('off'),plt.title("G0 " + str(pyrG0.shape[:2])),plt.imshow(cv.cvtColor(pyrG0, cv.COLOR_BGR2RGB))
    plt.subplot(342),plt.axis('off'),plt.title("G1 " + str(pyrG1.shape[:2])),plt.imshow(cv.cvtColor(pyrG1, cv.COLOR_BGR2RGB))
    plt.subplot(343),plt.axis('off'),plt.title("G2 " + str(pyrG2.shape[:2])),plt.imshow(cv.cvtColor(pyrG2, cv.COLOR_BGR2RGB))
    plt.subplot(344),plt.axis('off'),plt.title("G3 " + str(pyrG3.shape[:2])),plt.imshow(cv.cvtColor(pyrG3, cv.COLOR_BGR2RGB))

    plt.subplot(345),plt.axis('off'),plt.title("L0 " + str(pyrL0.shape[:2])),plt.imshow(cv.cvtColor(pyrL0, cv.COLOR_BGR2RGB))
    plt.subplot(346),plt.axis('off'),plt.title("L1 " + str(pyrL1.shape[:2])),plt.imshow(cv.cvtColor(pyrL1, cv.COLOR_BGR2RGB))
    plt.subplot(347),plt.axis('off'),plt.title("L2 " + str(pyrL2.shape[:2])),plt.imshow(cv.cvtColor(pyrL2, cv.COLOR_BGR2RGB))
    plt.subplot(348),plt.axis('off'),plt.title("L3 " + str(pyrL3.shape[:2])),plt.imshow(cv.cvtColor(pyrL3, cv.COLOR_BGR2RGB))

    plt.subplot(349),plt.axis('off'),plt.title("LaplaceRebuild " + str(rebuildG0.shape[:2])),plt.imshow(cv.cvtColor(rebuildG0, cv.COLOR_BGR2RGB))
    plt.subplot(3,4,10),plt.axis('off'),plt.title("LaplaceRebuild " + str(rebuildG1.shape[:2])),plt.imshow(cv.cvtColor(rebuildG1, cv.COLOR_BGR2RGB))
    plt.subplot(3,4,11),plt.axis('off'),plt.title("LaplaceRebuild " + str(rebuildG2.shape[:2])),plt.imshow(cv.cvtColor(rebuildG2, cv.COLOR_BGR2RGB))
    plt.subplot(3,4,12),plt.axis('off'),plt.title("LaplaceRebuild " + str(rebuildG3.shape[:2])),plt.imshow(cv.cvtColor(rebuildG3, cv.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filepath1 = r"img/Fig1001.png"
    filepath2 = r"img/Fig1002.png"
    filepath3 = r"img/Fig1003.png"
    filepath4 = r"img/LenaGauss.png"
    filepath5 = r"img/Fig0301.png"


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

    # 空间滤波器之自适应中值滤波器
    # ImageAdaptationMedFilter(filepath3)

    # 空间滤波器之双边滤波器
    # ImageBilateralFilter(filepath4)

    # 空间滤波之钝化掩蔽与高提升滤波
    # ImagePassivation(filepath4)

    # 拉普拉斯算子 Laplacian
    # ImageLaplacian(filepath1)

    # sobel算子
    # ImageSobel(filepath1)

    # scharr算子
    # ImageScharr(filepath1)

    # 高斯金字塔
    # ImageGaussianPyramid(filepath5)

    # 拉普拉斯金字塔
    ImageLaplacianPyramid(filepath5)
