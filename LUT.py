import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

def imageReversal(filepath):
    img = cv.imread(filepath, flags=1)
    h,w,ch = img.shape

    print("img = ",h,w,ch)

    timeBegin = cv.getTickCount()
    imgInv = np.empty((w,h,ch), np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(ch):
                imgInv[i][j][k] = 255 - img[i][j][k]
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Image invert by nested loop: {} sec".format(round(time, 4)))
    cv.imshow("imgInv",imgInv)
    cv.waitKey(0)

    timeBegin = cv.getTickCount()
    transTable = np.array([(255-i) for i in range(256)]).astype(np.uint8)
    invLUT = cv.LUT(img, transTable)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Image invert by cv.LUT: {} sec".format(round(time, 4)))
    cv.imshow("imgLUT",invLUT)
    cv.waitKey(0)

# 灰度级 如 （256//8 = 32个灰度级别）
def imageColor(filepath):
    gray = cv.imread(filepath, flags=0)
    h,w = gray.shape[:2]
    
    timeBegin = cv.getTickCount()
    imgGray32 = np.empty((w,h), np.uint8)
    for i in range(h):
        for j in range(w):
            imgGray32[i][j] = (gray[i][j]//8)*8
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Grayscale reduction by nested loop: {} sec".format(round(time, 4)))
    

    timeBegin = cv.getTickCount()
    table32 = np.array([(i//8)*8 for i in range(256)]).astype(np.uint8)
    gray32 = cv.LUT(gray, table32)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Grayscale reduction by cv.LUT: {} sec".format(round(time, 4)))

    
    table8 = np.array([(i//32)*32 for i in range(256)]).astype(np.uint8)
    gray8 = cv.LUT(gray, table8)
    
    

    cv.imshow("imgInv",imgGray32)
    cv.imshow("gray32",gray32)
    cv.imshow("gray8",gray8)
    cv.waitKey(0)




if __name__ == '__main__':
    filepath = "/home/calmcar/work/python/opencv_test/img/lena.jpg"
    # imageReversal(filepath)
    imageColor(filepath)