import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 图像的阈值处理

def ImageFixedThreshold(filepath1):
    # 生成灰度图像
    hImg, wImg = 512, 512
    img = np.zeros((hImg, wImg), np.uint8)

    cv.rectangle(img, (60, 60), (450, 320), (127, 127, 127), -1)  # 矩形填充
    cv.circle(img, (256, 256), 120, (205, 205, 205), -1)  # 圆形填充

    # 添加高斯噪声
    mu, sigma = 0.0, 20.0
    noiseGause = np.random.normal(mu, sigma, img.shape)
    imgNoise = np.add(img, noiseGause)
    imgNoise = np.uint8(cv.normalize(imgNoise, None, 0, 255, cv.NORM_MINMAX))

    # 阈值处理
    ret, imgBin1 = cv.threshold(imgNoise, 63, 255, cv.THRESH_BINARY)  # thresh = 63
    ret, imgBin2 = cv.threshold(imgNoise, 125, 255, cv.THRESH_BINARY)  # thresh = 125
    ret, imgBin3 = cv.threshold(imgNoise, 175, 255, cv.THRESH_BINARY)  # thresh = 175

    histCV = cv.calcHist([imgNoise], [0], None, [256], [0,256])

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. Noise Img"),plt.imshow(imgNoise, cmap='gray')
    plt.subplot(233, yticks=[]),plt.title("3. Gray hist"), plt.bar(range(256), histCV[:, 0]), plt.axis([0, 255, 0, np.max(histCV)])
    plt.subplot(234),plt.axis('off'),plt.title("4. threshold = 63"),plt.imshow(imgBin1, cmap='gray')
    plt.subplot(235),plt.axis('off'),plt.title("5. threshold = 125"),plt.imshow(imgBin2, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. threshold = 175"),plt.imshow(imgBin3, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageGlobalThreshold(filepath1):
    img = cv.imread(filepath1, flags=0)

    deltaT = 1   # 预定义值
    histCV = cv.calcHist([img], [0], None, [256], [0, 256])   # 灰度直方图
    grayScale = range(256)  # 灰度级[0, 256]
    totalPixels = img.shape[0] * img.shape[1]  # 像素总数

    tatalGray = np.dot(histCV[:, 0], grayScale) # 内积， 总和灰度值
    T = round(tatalGray/ totalPixels) # 平均灰度作为阈值初值

    while(True):  # 迭代基数按分割阈值
        numC1 = np.sum(histCV[:T, 0])  # C1 像素数量
        sumC1 = np.sum(histCV[:T, 0] * range(T))  # C1 灰度值总和

        numC2 = totalPixels - numC1    # C2 像素数量
        sumC2 = tatalGray - sumC1      # C2 灰度值总和

        T1 = round(sumC1 / numC1)    # C1 平均灰度
        T2 = round(sumC2 / numC2)    # C2 平均灰度

        Tnew = round((T1 + T2) / 2)  # 计算新的阈值

        print("T = {}, m1 = {}, m2 = {}, Tnew = {}".format(T, T1, T2, Tnew))

        if abs(T - Tnew) < deltaT :  # 等价于 T==Tnew
            break
        else:
            T = Tnew
        
    # 阈值处理
    ret, imgBin = cv.threshold(img, T, 255, cv.THRESH_BINARY)  # 阈值分割

    plt.figure(figsize=(9,3.3))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray')
    plt.subplot(132, yticks=[]),plt.title("2. Gray hist"), plt.bar(range(256), histCV[:, 0]), plt.axis([0, 255, 0, np.max(histCV)])
    plt.axvline(T, color='r', linestyle='--')  # 绘制阈值线
    plt.text(T + 5, 0.9*np.max(histCV), "T={}".format(T), fontsize=10)
    plt.subplot(133),plt.axis('off'),plt.title("3. Binary (T={})".format(T)),plt.imshow(imgBin, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageOTSU(filepath1):
    img = cv.imread(filepath1, flags=0)

    # OTSU 算法实现
    histCV = cv.calcHist([img], [0], None, [256], [0, 256])   # 灰度直方图
    scale = range(256)   # 灰度级[0, 255]
    totalPixels = img.shape[0] * img.shape[1]   # 像素总数
    totalGray = np.dot(histCV[:, 0], scale)   # 内积， 像素总和
    mG = totalGray / totalPixels  # 平均灰度
    icv = np.zeros(256)
    numFt, sumFt = 0, 0

    for t in range(256):
        numFt += histCV[t, 0]       # F(t)像素数量
        sumFt += histCV[t, 0] * t   # F(t)的灰度值总和
        pF = numFt / totalPixels    # F(t)的像素数占比
        mF = (sumFt / numFt) if numFt > 0 else 0    # F(t)的平均灰度
        numBt = totalPixels - numFt  # B(t)像素数量
        sumBt = totalGray - numFt    # B(t)的灰度值总和
        pB = numBt / totalPixels     # B(t)像素数占比
        mB = (sumBt / numBt) if numBt > 0 else 0    # B(t)的平均灰度
        icv[t] = pF * (mF - mG) ** 2 + pB * (mB - mG) ** 2   # 灰度t的类间方差
    
    maxIcv = max(icv)   # ICV的最大值
    maxIndex = np.argmax(icv)  # 最大值的索引， 即OTSU阈值

    _, imgBin = cv.threshold(img, maxIndex, 255, cv.THRESH_BINARY)   # 以maxIndex为最优阈值

    # 函数cv.threshold 实现OTSU算法      
    ret, imgOTSU = cv.threshold(img, 128, 255, cv.THRESH_OTSU)   # OTSU 阈值分割
    print("maxIndex = {}, retOTSU = {}".format(maxIndex, round(ret)))
    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray')
    plt.subplot(132),plt.axis('off'),plt.title("2. OTSU by ICV (T={})".format(maxIndex)),plt.imshow(imgBin, cmap='gray')
    plt.subplot(133),plt.axis('off'),plt.title("3. OTSU by OPENCV (T={})".format(round(ret))),plt.imshow(imgOTSU, cmap='gray')
    plt.tight_layout()
    plt.show()


def ImageAdaptiveThreshold(filepath1):
    img = cv.imread(filepath1, flags=0)

    # 自适应局部阈值处理
    binaryMean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 3)

    binaryGauss = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 3)

    # 参考方法 ： 自适应局部阈值处理
    ratio = 0.03
    imgBlur = cv.boxFilter(img, -1, (3, 3))  # 盒式滤波器， 均值平滑
    localThresh = img - (1.0 - ratio) * imgBlur
    binaryBox = np.ones_like(img) * 255      # 创建与img相同形状的白色图像
    binaryBox[localThresh < 0] = 0

    plt.figure(figsize=(9,3))
    plt.subplot(131),plt.axis('off'),plt.title("1. adaptive mean"),plt.imshow(binaryMean, cmap='gray')
    plt.subplot(132),plt.axis('off'),plt.title("2. adaptive Gauss"),plt.imshow(binaryGauss, cmap='gray')
    plt.subplot(133),plt.axis('off'),plt.title("3. adaptive local thresh"),plt.imshow(binaryBox, cmap='gray')
    plt.tight_layout()
    plt.show()


    


if __name__ == "__main__" :
    filepath1 = r"img/Lena.tif"
    filepath2 = r"img/Fig0301.png"

    # 固定阈值法
    # ImageFixedThreshold(filepath1)

    # 全局阈值法
    # ImageGlobalThreshold(filepath2)

    # 阈值处理之OTSU
    # ImageOTSU(filepath2)

    # 自适应阈值处理
    ImageAdaptiveThreshold(filepath2)
