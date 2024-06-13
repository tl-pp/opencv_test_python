import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 图像的直方图处理

def ImageHistogram(filepath1):
    img = cv.imread(filepath1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # OpenCV: cv.calcHist 计算灰度直方图
    histCV = cv.calcHist([gray], [0], None, [256], [0, 255]) 

    # Numpy: np.calcHist 计算灰度直方图
    histNP, bins = np.histogram(gray.flatten(), 256)

    print(histCV.shape, histNP.shape)
    print(histCV.max(), histNP.max())

    plt.figure(figsize=(9, 3))
    plt.subplot(131, yticks=[]), plt.axis([0, 255, 0, np.max(histCV)]), plt.title("1. Gray histogram NP")
    plt.bar(bins[:-1], histNP)
    plt.subplot(132, yticks=[]), plt.axis([0, 255, 0, np.max(histCV)]), plt.title("2. Gray histogram OpenCV")
    plt.bar(range(256), histCV[:, 0])

    # 计算和绘制彩色图像各个通道的直方图
    plt.subplot(133, yticks=[])
    plt.title("3. Color histogams OpenCV")
    color = ['b', 'g', 'r']
    for ch, col in enumerate(color):
        histCh = cv.calcHist([img], [ch], None, [256], [0, 255])
        plt.plot(histCh, color=col)
        plt.xlim([0,256])

    plt.tight_layout()
    plt.show()


def ImageHistogramEqualization(filepath1):
    gray = cv.imread(filepath1, flags=0)
    # gray = cv.multiply(gray, 0.6)
    histSrc = cv.calcHist([gray], [0], None, [256], [0, 255])

    # 直方图均衡化
    grayEqualize = cv.equalizeHist(gray)   # 直方图均衡化变换
    histEqual = cv.calcHist([grayEqualize], [0], None, [256], [0, 255])

    # 与直方图归一化比较
    grayNorm = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
    histNorm = cv.calcHist([grayNorm], [0], None, [256], [0, 255])

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232),plt.axis('off'),plt.title("2. Normalized"),plt.imshow(grayNorm, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233),plt.axis('off'),plt.title("3. Hist-equalize"),plt.imshow(grayEqualize, cmap='gray', vmin=0, vmax=255)

    plt.subplot(234, yticks=[]),plt.axis([0, 255, 0, np.max(histSrc)]),plt.title("4. Gray hist of src")
    plt.bar(range(256), histSrc[:, 0])

    plt.subplot(235, yticks=[]),plt.axis([0, 255, 0, np.max(histSrc)]),plt.title("5. Gray hist of normalized")
    plt.bar(range(256), histNorm[:, 0])

    plt.subplot(236, yticks=[]),plt.axis([0, 255, 0, np.max(histSrc)]),plt.title("4. Gray hist of equalized")
    plt.bar(range(256), histEqual[:, 0])

    plt.tight_layout()
    plt.show()




if __name__ == "__main__" :
    filepath1 = r"img/lenna.bmp"
    filepath2 = r"img/monarch.bmp"

    # 灰度图像和彩色图像的直方图
    # ImageHistogram(filepath1)

    # 直方图均衡化
    # ImageHistogramEqualization(filepath1)