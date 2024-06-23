import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 形态学图像处理

def ImageErodeAndDilate(filepath):
    img = cv.imread(filepath, flags=0)

    # 图像反转
    transtable = np.array([(255 - i) for i in range(256)]).astype(np.uint8)
    invLut = cv.LUT(img, transtable)

    _, imgBin = cv.threshold(invLut, 20, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法

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

    _, imgBin = cv.threshold(invLut, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法

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

    _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法

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

    _, binary = cv.threshold(invLut, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法

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


def ImageGrayScaleTophat(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBin1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法

    # 灰度顶帽算子后用OTSU最优阈值处理方法进行二值处理
    element = cv.getStructuringElement(cv.MORPH_RECT, (80, 80))       # 矩形结构元
    imgThat = cv.morphologyEx(img, cv.MORPH_TOPHAT, element)          # 灰度顶帽算子
    ret, imgBin2 = cv.threshold(imgThat, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值处理

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234),plt.axis('off'),plt.title("4. Tophat"),plt.imshow(imgThat, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. Original binary"),plt.imshow(imgBin1, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. Tophat binary"),plt.imshow(imgBin2, cmap='gray')

    h = np.arange(0, img.shape[1])
    w = np.arange(0, img.shape[0])
    xx, yy = np.meshgrid(h, w)        # 转换为网格点集（二维数组）
    ax1 = plt.subplot(232, projection='3d')
    ax1.plot_surface(xx, yy, img, cmap='coolwarm')
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_zticks([])
    ax1.set_title("2. Original grayscale")

    ax2 = plt.subplot(235, projection='3d')
    ax2.plot_surface(xx, yy, imgThat, cmap='coolwarm')
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_zticks([])
    ax2.set_title("5. Tophat grayscale")
    plt.tight_layout()
    plt.show()


def ImageGrayScaleBlackhat(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBin1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法

    # 底帽运算
    r = 80    # 特征尺寸， 由目标大小确定
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (r, r))    # 圆形结构元
    imgBhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, element)      # 底帽运算
    _, imgBin2 = cv.threshold(imgBhat, 20, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 闭运算去除圆环噪点
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
    imgSegment = cv.morphologyEx(imgBin2, cv.MORPH_CLOSE, element)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234),plt.axis('off'),plt.title("4. Blackhat"),plt.imshow(imgBhat, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. Original binary"),plt.imshow(imgBin1, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. Blackhat binary"),plt.imshow(imgSegment, cmap='gray')

    h = np.arange(0, img.shape[1])
    w = np.arange(0, img.shape[0])
    xx, yy = np.meshgrid(h, w)        # 转换为网格点集（二维数组）
    ax1 = plt.subplot(232, projection='3d')
    ax1.plot_surface(xx, yy, img, cmap='coolwarm')
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_zticks([])
    ax1.set_title("2. Original grayscale")

    ax2 = plt.subplot(235, projection='3d')
    ax2.plot_surface(xx, yy, imgBhat, cmap='coolwarm')
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_zticks([])
    ax2.set_title("5. Blackhat grayscale")
    plt.tight_layout()
    plt.show()


def ImageBoundaryExtraction(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法
    # _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # 3x3
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))       # 矩形结构元
    imgErode1 = cv.erode(imgBin, element)
    imgBound1 = imgBin - imgErode1   # 图像边界提取

    #9x9
    element = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))       # 矩形结构元
    imgErode2 = cv.erode(imgBin, element)
    imgBound2 = imgBin - imgErode2   # 图像边界提取

    plt.figure(figsize=(9,3.3))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Boundary extraction (3,3)"),plt.imshow(imgBound1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Boundary extraction (9,9)"),plt.imshow(imgBound2, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageLineExtraction(filepath):
    img = cv.imread(filepath, flags=0) 
    # _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)   # 二值处理 OTSU最优阈值处理方法
    _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    h,w = imgBin.shape[0],  imgBin.shape[1]

    # 提取水平线
    hline = cv.getStructuringElement(cv.MORPH_RECT, ((w//16), 1), (-1, -1))      # 水平结构元
    imgOpenHline = cv.morphologyEx(imgBin, cv.MORPH_OPEN, hline)                 # 开运算提取水平结构
    imgHline = cv.bitwise_not(imgOpenHline)     # 恢复白色背景

    # 提取垂直线
    vline = cv.getStructuringElement(cv.MORPH_RECT, (1, (h//16)), (-1, -1))      # 垂直结构元
    imgOpenVline = cv.morphologyEx(imgBin, cv.MORPH_OPEN, vline)                 # 开运算提取垂直结构
    imgVline = cv.bitwise_not(imgOpenVline)     # 恢复白色背景

    # 删除水平线和垂直线
    lineRemoved = imgBin - imgOpenHline           # 删除水平线（白底为0）
    lineRemoved = lineRemoved - imgOpenVline      # 删除垂直线
    lineRebuild = cv.bitwise_not(lineRemoved)     # 恢复白色背景 

    plt.figure(figsize=(9,3))
    plt.subplot(141),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(142),plt.axis('off'),plt.title("2. Horizontal line"),plt.imshow(imgHline, cmap='gray', vmin=0, vmax=255)
    plt.subplot(143),plt.axis('off'),plt.title("3. Vertical line"),plt.imshow(imgVline, cmap='gray', vmin=0, vmax=255)
    plt.subplot(144),plt.axis('off'),plt.title("3. H/V line removed"),plt.imshow(lineRebuild, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

def ImageBoundaryClear(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV )  # 二值处理

    imgBinInv = cv.bitwise_not(imgBin)     # 二值图像的补集 (白色背景)， 用于构造标记图像

    # 构造标记图像：
    F0 = np.zeros(img.shape, np.uint8)     # 边界为imgBin, 其他全黑
    F0[:, 0] = imgBin[:, 0]
    F0[:, -1] = imgBin[:, -1]
    F0[0, :] = imgBin[0, :]
    F0[-1, :] = imgBin[-1, :]

    # 形态学重建
    Flast = F0.copy()   # F(k) 初值
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    iter = 0
    while True:
        dilateF = cv.dilate(Flast, element)
        Fnew = cv.bitwise_and(dilateF, imgBin)
        if (Fnew == Flast).all() :     # 收敛判断 F(k+1)=F(k)?
            break                      # 迭代结束
        else:
            Flast = Fnew.copy()
        
        iter += 1                      # 迭代次数
        if iter == 5:
            imgF1 = Fnew               # 显示中间结果
        elif iter == 50:
            imgF50 = Fnew              # 显示中间结果
    print("iter = ", iter)
    imgRebuild = cv.bitwise_and(imgBin, cv.bitwise_not(Fnew))   # 计算边界清楚后的图像

    plt.figure(figsize=(9,5.6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. Template ($I^c$)"),plt.imshow(imgBin, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. Initial marker"),plt.imshow(imgF1, cmap='gray')
    plt.subplot(234),plt.axis('off'),plt.title("4. Marker iter=50"),plt.imshow(imgF50, cmap='gray')
    plt.subplot(235),plt.axis('off'),plt.title("5. Final marker"),plt.imshow(Fnew, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. Rebuild img"),plt.imshow(imgRebuild, cmap='gray')
    plt.tight_layout()
    plt.show()

def ImageHoleFilling(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV )  # 二值处理

    imgBinInv = cv.bitwise_not(imgBin)     # 二值图像的补集 (白色背景)， 用于构造标记图像

    # 构造标记图像
    F0 = np.zeros(imgBinInv.shape, np.uint8)     # 边界为imgBin, 其他全黑
    F0[:, 0] = imgBinInv[:, 0]
    F0[:, -1] = imgBinInv[:, -1]
    F0[0, :] = imgBinInv[0, :]
    F0[-1, :] = imgBinInv[-1, :]

    # 形态学重建
    Flast = F0.copy()  # F(k)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))   # 十字型结构元
    iter = 0
    while True:
        dilateF = cv.dilate(Flast, element)          # 标记图像膨胀
        Fnew = cv.bitwise_and(dilateF, imgBinInv)    # 以原始图像的补集作为模板约束重建
        if (Fnew == Flast).all() :     # 收敛判断 F(k+1)=F(k)?
            break                      # 迭代结束
        else:
            Flast = Fnew.copy()
        iter += 1                      # 迭代次数
        if iter == 2:
            imgF1 = Fnew               # 显示中间结果
        elif iter == 100:
            imgF100 = Fnew              # 显示中间结果
    print("iter = ", iter)
    imgRebuild = cv.bitwise_not(Fnew)  # F(k) 的补集是孔洞填充的重建结果
    
    plt.figure(figsize=(9,5.6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. Template ($I^c$)"),plt.imshow(imgBinInv, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. Initial marker"),plt.imshow(imgF1, cmap='gray')
    plt.subplot(234),plt.axis('off'),plt.title("4. Marker iter=100"),plt.imshow(imgF100, cmap='gray')
    plt.subplot(235),plt.axis('off'),plt.title("5. Final marker"),plt.imshow(Fnew, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. Rebuild img"),plt.imshow(imgRebuild, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageFloodFill(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBinInv = cv.threshold(img, 205, 255, cv.THRESH_BINARY)  # 二值处理(白色背景)
    imgBin = cv.bitwise_not(imgBinInv)     # 二值图像的补集 (黑色背景)， 填充基准

    h,w = imgBin.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)   # 掩摸图像比图像宽2个像素，高2个像素
    imgFloodfill = imgBin.copy()            # 输出孔洞图像，返回填充孔洞
    cv.floodFill(imgFloodfill, mask, (0,0), newVal=225)  # 从背景像素原点(0, 0)点开始
    imgRebuild = cv.bitwise_and(imgBinInv, imgFloodfill)

    plt.figure(figsize=(9,3.3))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Filled holes"),plt.imshow(imgFloodfill, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Rebuild image"),plt.imshow(imgRebuild, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageSkeleton(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBin = cv.threshold(img, 127, 255, cv.THRESH_BINARY)  # 二值处理

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))   # 十字型结构元
    skeketon = np.zeros(imgBin.shape, np.uint8)    # 创建空骨架图
    Fk = cv.erode(imgBin, element)                 # 标记图像Fk的初值
    while True:
        imgOpen = cv.morphologyEx(Fk, cv.MORPH_OPEN, element)    # 开运算
        subSkel = cv.subtract(Fk, imgOpen)                       # 获得本次骨架的子集
        skeketon = cv.bitwise_or(skeketon, subSkel)                    # 将删除的像素添加到骨架图
        if cv.countNonZero(Fk) == 0:      # 收敛判断
            break
        else:
            Fk = cv.erode(Fk, element)    # 更新 Fk

    skeketon = cv.dilate(skeketon, element)  # 膨胀以便显示， 非必需步骤
    result = cv.bitwise_xor(img, skeketon)

    plt.figure(figsize=(9,3.3))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(imgBin, cmap='gray')
    plt.subplot(132),plt.axis('off'),plt.title("2. Skeketon"),plt.imshow(cv.bitwise_not(skeketon), cmap='gray')
    plt.subplot(133),plt.axis('off'),plt.title("3. Stacked"),plt.imshow(result, cmap='gray')
    plt.tight_layout()
    plt.show()



def morphRebuild(F0, template):
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))  # 十字型结构元
    Flast = F0                  # F0, 重建标记
    while True:
        dilateF = cv.dilate(Flast, element)        # 标记图像膨胀
        Fnew = cv.bitwise_and(dilateF, template)   # 模板约束重建
        if (Fnew == Flast).all():   # 收敛判断， F(k+1) = F(k)?
            break
        else:
            Flast = Fnew
    imgRebuild = Fnew
    return imgRebuild

def ImageParticleSeparation(filepath):
    img = cv.imread(filepath, flags=0) 
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理

    imgBinInv = cv.bitwise_not(imgBin)

    # (1)垂直特征结构元
    element = cv.getStructuringElement(cv.MORPH_RECT, (1, 60))    # 垂直特征结构元
    imgErode = cv.erode(imgBin, element)  # 腐蚀结果作为标记图像
    imgRebuild1 = morphRebuild(imgErode, imgBin)
    imgDuall = cv.bitwise_and(imgBin, cv.bitwise_not(imgRebuild1, imgErode))

    # (2)水平特征结构元
    element = cv.getStructuringElement(cv.MORPH_RECT, (60, 1))    # 水平特征结构元
    imgErode = cv.erode(imgBin, element)  # 腐蚀结果作为标记图像
    imgRebuild2 = morphRebuild(imgErode, imgBin)

    # (3)圆形结构元
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
    imgErode = cv.erode(imgBin, element)  # 腐蚀结果作为标记图像
    imgRebuild3 = morphRebuild(imgErode, imgBin)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. Binary"),plt.imshow(imgBin, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. Initial marker"),plt.imshow(imgErode, cmap='gray')
    plt.subplot(234),plt.axis('off'),plt.title("4. Rebuild (1, 60)"),plt.imshow(imgRebuild1, cmap='gray')
    plt.subplot(235),plt.axis('off'),plt.title("5. Rebuild (60, 1)"),plt.imshow(imgRebuild2, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. Rebuild (60, 60)"),plt.imshow(imgRebuild3, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageParticleDetermination(filepath):
    img = cv.imread(filepath, flags=0)
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("Original"),plt.imshow(img, cmap='gray')

    # 用不同半径圆形结构元进行开运算
    rList= [14, 21, 28, 35, 42]
    for i in range(5):
        size = rList[i] * 2 + 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))   # 圆形结构元
        imgOpen = cv.morphologyEx(imgBin, cv.MORPH_ELLIPSE, element)
        plt.subplot(2, 3, i + 2), plt.title("Opening (r={})".format(rList[i])), plt.imshow(cv.bitwise_not(imgOpen), cmap='gray'), plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 计算圆形直径的半径分布
    maxSize = 42
    sumSurf = np.zeros(maxSize)
    deltaSum = np.zeros(maxSize)
    for r in range(5, maxSize):
        size = r * 2  + 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        imgOpen = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        sumSurf[r] = np.concatenate(imgOpen).sum()
        deltaSum[r] = sumSurf[r-1] - sumSurf[r]
        print(r, sumSurf[r], deltaSum[r])

    r = range(maxSize)
    plt.figure(figsize=(6,4))
    plt.plot(r[6:], deltaSum[6:], 'b-o')
    plt.title("Delta of surface area")
    plt.yticks([])
    plt.show()


def ImageCornerDetection(filepath):
    img = cv.imread(filepath, flags=1)
    imgSign = img.copy()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 边缘检测
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    imgEdge = cv.morphologyEx(imgGray, cv.MORPH_GRADIENT, element)    # 形态学梯度

    # 构造 9x9 结构元, 包括十字形结构元， 菱形结构元， 方形结构元和x刑结构元
    cross = cv.getStructuringElement(cv.MORPH_CROSS, (9, 9))          # 结构十字形结构元
    square = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))          # 结构方形结构元
    xShape = cv.getStructuringElement(cv.MORPH_CROSS, (9, 9))         # 构造X形结构元
    diamond = cv.getStructuringElement(cv.MORPH_CROSS, (9, 9))        # 构造菱形结构元
    diamond[1, 1] = diamond[3, 3] = 1
    diamond[1, 3] = diamond[3, 1] = 1
    print(diamond)

    imgDilate1 = cv.dilate(imgGray, cross)        # 用十字形结构元膨胀原始图像
    imgErode1 = cv.erode(imgDilate1, diamond)     # 用菱形结构元腐蚀图像

    imgDilate2 = cv.dilate(imgGray, xShape)       # 用X形结构元膨胀原始图像
    imgErode2 = cv.erode(imgDilate2, square)      # 用方形结构元腐蚀图像

    imgDiff = cv.absdiff(imgErode2, imgErode1)    # 将两幅闭运算的图像相减获得角点
    retval, thresh = cv.threshold(imgDiff, 40, 255, cv.THRESH_BINARY)    # 二值处理

    # 在原始图像上用半径为5的圆圈标记角点
    for j in range(thresh.size):
        y = int(j / thresh.shape[0])
        x = int(j % thresh.shape[0])
        if (thresh[x, y] == 255):
            cv.circle(imgSign, (y, x), 5, (255, 0, 255))

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. Morph edge"),plt.imshow(cv.bitwise_not(imgEdge), cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Morph corner"),plt.imshow(cv.cvtColor(imgSign, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    filepath1 = r"img/Fig1201.png"
    filepath2 = r"img/Fig0703.png"
    filepath3 = r"img/Fig1202.png"
    filepath4 = r"img/Fig1101.png"
    filepath5 = r"img/Fig1203.png"
    filepath6 = r"img/Fig1204.png"
    filepath7 = r"img/Fig0801.png"
    filepath8 = r"img/Fig1001.png"
    filepath9 = r"img/Fig1205.png"
    filepath10 = r"img/Fig1206.png"
    filepath11 = r"img/Fig1207.png"
    filepath12 = r"img/Fig1208.png"
    filepath13 = r"img/Fig1209.png"

    # 形态学之腐蚀与膨胀
    # ImageErodeAndDilate(filepath1)

    # 形态学之开运算与闭运算
    # ImageOpenAndClose(filepath1)

    # 形态学之梯度运算
    # ImageGradient(filepath2)

    # 用击中-击不中变换进行特征识别
    # ImageHitMiss(filepath3)

    # 灰度形态学运算
    # ImageGrayScale(filepath4)

    # 灰度顶帽算子校正光照影响
    # ImageGrayScaleTophat(filepath5)

    # 灰度顶帽算子校正光照影响
    # ImageGrayScaleBlackhat(filepath6)

    # 形态学算法之边界提取
    # ImageBoundaryExtraction(filepath7)

    # 形态学算法之直线提取
    # ImageLineExtraction(filepath8)

    # 形态学重建之边界清除
    # ImageBoundaryClear(filepath9)

    # 形态学重建之孔洞填充
    # ImageHoleFilling(filepath9)

    # 泛洪填充算法实现孔洞填充
    # ImageFloodFill(filepath9)

    # 形态学重建之骨架提取
    # ImageSkeleton(filepath10)

    # 形态学之粒径分离
    # ImageParticleSeparation(filepath11)

    # 形态学之粒度测定
    # ImageParticleDetermination(filepath12)

    # 形态学之边缘和角点检测
    ImageCornerDetection(filepath13)