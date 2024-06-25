import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 图像变换、重建与复原

def ImagePolarTransform(filepath):
    img = cv.imread(filepath)
    h, w = img.shape[:2]

    cx, cy = int(w/2), int(h/2)    # 以图像中心点作为变换中心
    maxR = max(cx, cy)             # 最大变换半径
    imgPolar = cv.linearPolar(img, (cx, cy), maxR, cv.INTER_LINEAR)
    imgPR = cv.rotate(imgPolar, cv.ROTATE_90_COUNTERCLOCKWISE)
    imgRebuild = np.hstack((imgPR[:, w//2:], imgPR[:,:w//2]))

    print(img.shape, imgRebuild.shape)

    plt.figure(figsize=(9,3.5))
    plt.subplot(131),plt.axis('off'),plt.title("1. Original"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. Polar Transform"),plt.imshow(cv.cvtColor(imgPolar, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. Polar Rebuild"),plt.imshow(cv.cvtColor(imgRebuild, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


def ImageHough(filepath):
    img = cv.imread(filepath, flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hImg, wImg = gray.shape

    # (1) Canny 算子进行边缘检测， TL、TH为低阈值、高阈值
    TL, ratio = 60, 3         # ratio = TH/TL
    imgGauss = cv.GaussianBlur(gray, (5, 5), 0)
    imgCanny = cv.Canny(imgGauss, TL, TL*ratio)

    # (2) 霍夫变换进行直线检测
    imgEdge1 = cv.convertScaleAbs(img, alpha=0.25, beta=192)
    lines = cv.HoughLines(imgCanny, 1, np.pi/180, threshold=100)     # (n, 1, 2)
    print("cv.HoughLines: ", lines.shape)    # 每行元素（i,1,:）表示直线参数  rho和theta
    for i in range(lines.shape[0]//2):       # 绘制部分检测直线
        rho, theta = lines[i, 0, :]          # lines每行两个元素
        if (theta < (np.pi/4)) or (theta>(3*np.pi/4)):   # 直线与图像上下相交
            pt1 = (int(rho/np.cos(theta)), 0)                            #  (x,0) 直线与顶侧的交点
            pt2 = (int(rho - hImg*np.sin(theta)/np.cos(theta)), hImg)    # （x,h） 直线与底侧的交点
            cv.line(imgEdge1, pt1, pt2, (255, 127, 0), 2)      # 绘制直线
        else:                                             # 直线与图像左右相交
            pt1 = (0, int(rho/np.sin(theta)))                            #  (0,y) 直线与左侧的交点
            pt2 = (wImg, int((rho - wImg*np.cos(theta))/ np.sin(theta))) #  (w,y) 直线与右侧的交点
            cv.line(imgEdge1, pt1, pt2, (255, 0, 127), 2)      # 绘制直线
        print("rho = {}, theta = {:.1f}".format(rho, theta))

    # (3) 积累概率霍夫变换
    imgEdge2 = cv.convertScaleAbs(img, alpha=0.25, beta=192)
    minLineLength = 30                 # 检测直线的最小长度
    maxLineGap = 10                    # 直线上像素的最大间隔
    lines = cv.HoughLinesP(imgCanny, 1, np.pi/180, 60, minLineLength, maxLineGap)   # lines: (n, 1, 4)
    print("cv.HoughLinesP: ", lines.shape)    # 每行元素（i,1,:）表示参数  x1, y1, x2, y2
    for line in lines:
        x1, y1, x2, y2 = line[0]              # 返回值每行是1个4元组， 表示直线端点(x1, y1, x2, y2)
        cv.line(imgEdge2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print("(x1, y1) = ({}, {}), (x2, y2) = ({}, {})".format(x1, y1, x2, y2))

    plt.figure(figsize=(9,3.3))
    plt.subplot(131),plt.axis('off'),plt.title("1. Canny edges"),plt.imshow(cv.bitwise_not(imgCanny), cmap='gray')
    plt.subplot(132),plt.axis('off'),plt.title("2. cv.HoughLines"),plt.imshow(cv.cvtColor(imgEdge1, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. cv.HoughLinesP"),plt.imshow(cv.cvtColor(imgEdge2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()








if __name__ == "__main__":
    filepath1 = r"img/Fig1301.png"
    filepath2 = r"img/Fig1201.png"

    # 极坐标中的环形图案和文字校正
    # ImagePolarTransform(filepath1) 

    # 霍夫变换直线检测
    ImageHough(filepath2)