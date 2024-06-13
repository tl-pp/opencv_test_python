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

def Imagelog(filepath1):
    gray = cv.imread(filepath1, flags = 0)

    fft = np.fft.fft2(gray)  # 傅里叶变换
    fft_shift = np.fft.fftshift(fft) # 将低频部分移动到图像中心
    amp = np.abs(fft_shift)  # 傅里叶变换的频谱
    ampNorm = np.uint8(cv.normalize(amp, None, 0, 255, cv.NORM_MINMAX))  # 归一化
    ampLog = np.abs(np.log(1.0 + np.abs(fft_shift)))  # 对数变换， c = 1
    ampLogNorm = np.uint8(cv.normalize(ampLog, None, 0, 255, cv.NORM_MINMAX))

    plt.figure(figsize=(9,3.4))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. FFT spectrum"),plt.imshow(ampNorm, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. LogTrans of FFT"),plt.imshow(ampLogNorm, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

def ImageGamma(filepath1):
    gray = cv.imread(filepath1, flags = 0)

    c = 1
    gammas = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
    fig = plt.figure(figsize=(9, 5.5))
    for i in range(len(gammas)):
        ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[])
        img_gamma = c * np.power(gray, gammas[i]) # 伽马变换
        ax.imshow(img_gamma, cmap='gray')
        if(gammas[i] == 1.0):
            ax.set_title("1. Original gamma=1.0")
        else:
            ax.set_title(f"{i+1}. $gamma={gammas[i]}$")

    plt.tight_layout()
    plt.show()

def ImageSegmentedStretching(filepath1):
    gray = cv.imread(filepath1, flags = 0)

    # 拉伸控制点
    r1, s1 = 128, 64   # 第一个转折点（r1, s1）
    r2, s2 = 192, 224  # 第二个转折点（r2, s2）

    # LUT函数快速查表法实现对比度拉伸
    luTable = np.zeros(256)
    for i in range(256):
        if i < r1:
            luTable[i] = (s1 / r1) * i
        elif i < r2:
            luTable[i] = (s2 - s1) / (r2 - r1) * (i - r1) + s1
        else:
            luTable[i] = ((s2 - 255.0) / (r2 - 255.0)) * (i - r2) + s2
    imgSLT = np.uint8(cv.LUT(gray, luTable))

    print(luTable)

    plt.figure(figsize=(9,3))
    plt.subplot(131), plt.axis('off'), plt.title("1. Original"), plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("2. s=T(r)")
    r = [0, r1, r2, 255]
    s = [0, s1, s2, 255]
    plt.plot(r, s)
    plt.axis([0, 256, 0, 256])
    plt.text(128, 40, "(r1, s1)", fontsize=10)
    plt.text(128, 220, "(r2, s2)", fontsize=10)
    plt.xlabel("r, Input value")
    plt.ylabel("s, Output value")
    plt.subplot(133), plt.axis('off'), plt.title("3. Stretched"), plt.imshow(imgSLT, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageGrayscaleLayering(filepath1):
    gray = cv.imread(filepath1, flags = 0)
    width, height = gray.shape[:2]

    # 方案1： 二值变化灰度分层
    a, b = 155, 245  # 突出[a, b]区间的灰度
    binLayer = gray.copy()
    binLayer[(binLayer[:,:] < a) | (binLayer[:,:] > b)] = 0
    binLayer[(binLayer[:,:] >= a) & (binLayer[:,:] <= b)] = 245 # 灰度级窗口，白色。 其他，黑色

    # 方案2： 增强选择的灰度窗口
    a, b = 155, 245
    winlayer = gray.copy()
    winlayer[(winlayer[:,:] >= a) & (winlayer[:,:] <= b)] = 245  # 灰度级窗口，白色。 其他不变

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132),plt.axis('off'),plt.title("2. Binary layered"),plt.imshow(binLayer, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133),plt.axis('off'),plt.title("3. Window layered"),plt.imshow(winlayer, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


def ImageBitPlane(filepath1):
    gray = cv.imread(filepath1, flags = 0)
    width, height = gray.shape[:2]

    bitLayer = np.zeros((8, height, width), np.uint(8))
    bitLayer[0] = cv.bitwise_and(gray, 1)   # 按位与 0000 0001
    bitLayer[1] = cv.bitwise_and(gray, 2)   # 按位与 0000 0010
    bitLayer[2] = cv.bitwise_and(gray, 4)   # 按位与 0000 0100
    bitLayer[3] = cv.bitwise_and(gray, 8)   # 按位与 0000 1000
    bitLayer[4] = cv.bitwise_and(gray, 16)   # 按位与 0001 0000
    bitLayer[5] = cv.bitwise_and(gray, 32)   # 按位与 0010 0000
    bitLayer[6] = cv.bitwise_and(gray, 64)   # 按位与 0100 0000
    bitLayer[7] = cv.bitwise_and(gray, 128)   # 按位与 1000 0000

    plt.figure(figsize=(9,8))
    plt.subplot(331),plt.axis('off'),plt.title("1. gray img"),plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    for bit in range(8):
        print(bit)
        plt.subplot(3,3,9-bit),plt.axis('off'),plt.title(f"{9-bit}. {(1 << bit):08b}"),plt.imshow(bitLayer[bit], cmap='gray')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__" :
    filepath1 = r"img/lenna.bmp"

    # 图像翻转变换
    # ImageGrayscaleFlip(filepath1)

    # 图像线性变换
    # ImageGrayLinearMapping(filepath1)

    # 图像直方图归一化
    # ImageHistogramNormalization(filepath1)

    # 对数变换
    # Imagelog(filepath1)

    # 伽马变换
    # ImageGamma(filepath1)

    # 分段线性变换 对比度拉伸
    # ImageSegmentedStretching(filepath1)

    # 分段线性变换 灰度级分层
    # ImageGrayscaleLayering(filepath1)

    # 比特平面
    # ImageBitPlane(filepath1)

    



