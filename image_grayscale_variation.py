import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

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



if __name__ == "__main__" :
    filepath1 = r"img/lenna.bmp"
    filepath2 = r"img/remap.png"

    ImageGrayscaleFlip(filepath1)