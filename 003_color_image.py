import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

#灰度图像转为伪彩色图像
def gray2ColorImage(filepath):
    gray = cv.imread(filepath, flags = 0)

    # 伪彩色处理
    pseudo1 = cv.applyColorMap(gray, colormap=cv.COLORMAP_HOT)
    pseudo2 = cv.applyColorMap(gray, colormap=cv.COLORMAP_PINK)
    pseudo3 = cv.applyColorMap(gray, colormap=cv.COLORMAP_RAINBOW)
    pseudo4 = cv.applyColorMap(gray, colormap=cv.COLORMAP_HSV)
    pseudo5 = cv.applyColorMap(gray, colormap=cv.COLORMAP_TURBO)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. GRAY"),plt.imshow(gray, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. COLORMAP_HOT"),plt.imshow(cv.cvtColor(pseudo1, cv.COLOR_BGR2RGB))
    plt.subplot(233),plt.axis('off'),plt.title("3. COLORMAP_PINK"),plt.imshow(cv.cvtColor(pseudo2, cv.COLOR_BGR2RGB))
    plt.subplot(234),plt.axis('off'),plt.title("4. COLORMAP_RAINBOW"),plt.imshow(cv.cvtColor(pseudo3, cv.COLOR_BGR2RGB))
    plt.subplot(235),plt.axis('off'),plt.title("5. COLORMAP_HSV"),plt.imshow(cv.cvtColor(pseudo4, cv.COLOR_BGR2RGB))
    plt.subplot(236),plt.axis('off'),plt.title("6. COLORMAP_TURBO"),plt.imshow(cv.cvtColor(pseudo5, cv.COLOR_BGR2RGB)) 
    plt.tight_layout()
    plt.show()

# 彩色风格滤镜
def FilterColorImage(filepath):
    img = cv.imread(filepath, flags = 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("origin"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    # 由matplotlib 构造自定义色彩映射表
    cmList = ["cm.copper", "cm.hot", "cm.YlOrRd", "cm.rainbow", "cm.prism"]
    for i in range(len(cmList)):
        cmMap = eval(cmList[i])(np.arange(256))
        # RGB(matplotlib)->BGR(opnecv)
        lutC3 = np.zeros((1,256,3)) # BGR(opencv)
        lutC3[0,:,0] = np.array(cmMap[:,2]*255).astype("uint8") # B cmHot[:2]
        lutC3[0,:,1] = np.array(cmMap[:,1]*255).astype("uint8") # G cmHot[:1]
        lutC3[0,:,2] = np.array(cmMap[:,0]*255).astype("uint8") # R cmHot[:0]

        cmLUTC3 = cv.LUT(img, lutC3).astype("uint8")
        plt.subplot(2, 3, i+2),plt.axis('off'),plt.title("{}. {}".format(i+2, cmList[i])),plt.imshow(cv.cvtColor(cmLUTC3, cv.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

# 调节色彩平衡
def AdjustColorBalance(filepath):
    img = cv.imread(filepath, flags = 1)

    # 生成单通道LUT，形状为（256,）
    maxG = 128  # 修改颜色通道最大值， 0 <= maxG <=255
    lutHalf = np.array([int(i * maxG / 255) for i in range(256)]).astype("uint8")
    lutEqual = np.array([i for i in range(256)]).astype("uint8")
    # 构造多通道LUT，形状为（1，256，3）
    lut3HalfB = np.dstack((lutHalf,lutEqual,lutEqual))  # B通道颜色衰减
    lut3HalfG = np.dstack((lutEqual,lutHalf,lutEqual))  # G通道颜色衰减
    lut3HalfR = np.dstack((lutEqual,lutEqual,lutHalf))  # R通道颜色衰减

    blendHalfB = cv.LUT(img,lut3HalfB)
    blendHalfG = cv.LUT(img,lut3HalfG)
    blendHalfR = cv.LUT(img,lut3HalfR)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("origin"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(234),plt.axis('off'),plt.title("1. B_ch half decayed"),plt.imshow(cv.cvtColor(blendHalfB, cv.COLOR_BGR2RGB))
    plt.subplot(235),plt.axis('off'),plt.title("2. G_ch half decayed"),plt.imshow(cv.cvtColor(blendHalfG, cv.COLOR_BGR2RGB))
    plt.subplot(236),plt.axis('off'),plt.title("3. R_ch half decayed"),plt.imshow(cv.cvtColor(blendHalfR, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def AdjustHSV(filepath):
    img = cv.imread(filepath, flags = 1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 生成单通道LUT，形状为（256,）
    k = 0.6
    lutWeaken = np.array([int(k*i) for i in range(256)]).astype("uint8")
    lutEqual = np.array([i for i in range(256)]).astype("uint8")
    lutRaisen = np.array([int(255*(1-k) + k*i) for i in range(256)]).astype("uint8")

    print("lutWeaken = ", lutWeaken)
    print("lutRaisen = ", lutRaisen)

    # 构造多通道LUT，调节饱和度
    lutSWeaken = np.dstack((lutEqual,lutWeaken,lutEqual))  #衰减
    lutSRaisen = np.dstack((lutEqual,lutRaisen,lutEqual))  #增强
    # 构造多通道LUT，调节明度
    lutVWeaken = np.dstack((lutEqual,lutEqual,lutWeaken))  #衰减
    lutVRaisen = np.dstack((lutEqual,lutEqual,lutRaisen))  #增强

    blendSWeaken = cv.LUT(hsv,lutSWeaken)
    blendSRaisen = cv.LUT(hsv,lutSRaisen)
    blendVWeaken = cv.LUT(hsv,lutVWeaken)
    blendVRaisen = cv.LUT(hsv,lutVRaisen)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("origin"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(232),plt.axis('off'),plt.title("1. Saturation weaken"),plt.imshow(cv.cvtColor(blendSWeaken, cv.COLOR_HSV2RGB))
    plt.subplot(233),plt.axis('off'),plt.title("2. Saturation raisen"),plt.imshow(cv.cvtColor(blendSRaisen, cv.COLOR_HSV2RGB))

    plt.subplot(234),plt.axis('off'),plt.title("origin"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(235),plt.axis('off'),plt.title("3. Value weaken"),plt.imshow(cv.cvtColor(blendVWeaken, cv.COLOR_HSV2RGB))
    plt.subplot(236),plt.axis('off'),plt.title("4. Value raisen"),plt.imshow(cv.cvtColor(blendVRaisen, cv.COLOR_HSV2RGB))
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    filepath = "/home/calmcar/work/python/opencv_test/img/lenna.bmp"
    # gray2ColorImage(filepath)
    # FilterColorImage(filepath)
    # AdjustColorBalance(filepath)
    AdjustHSV(filepath)