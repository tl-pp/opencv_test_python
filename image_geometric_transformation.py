import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 图像的几何变换

def ImageTranslation(filepath1):
    img = cv.imread(filepath1)
    height, width = img.shape[:2]
    print("height,width = ",height, width)

    dx, dy = 100, 50
    MAT = np.float32([[1,0,dx], [0,1,dy]])  # 构造平移变换矩阵
    
    imgTrans1 = cv.warpAffine(img,MAT, (width,height))
    imgTrans2 = cv.warpAffine(img,MAT, (601,401), borderValue=(0,0,0))

    plt.figure(figsize=(9,3.2))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. translation 1"),plt.imshow(cv.cvtColor(imgTrans1, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. translation 2"),plt.imshow(cv.cvtColor(imgTrans2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def ImageResize(filepath1):
    img = cv.imread(filepath1)
    height, width = img.shape[:2]
    print("height,width = ",height, width)
    
    imgResize1 = cv.resize(img, (600,480))
    imgResize2 = cv.resize(img, None, fx=1.2, fy=0.8, interpolation=cv.INTER_CUBIC)

    plt.figure(figsize=(9,3.2))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. resize 1"),plt.imshow(cv.cvtColor(imgResize1, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. resize 2"),plt.imshow(cv.cvtColor(imgResize2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def ImageRotate(filepath1):
    img = cv.imread(filepath1)
    height, width = img.shape[:2]
    print("height,width = ",height, width)

    # 以原点为中心旋转
    x0, y0 = 0, 0  # 左上顶点
    theta, scale = 30, 1.0 # 逆时针旋转30°， 缩放系数1.0
    MAR0 = cv.getRotationMatrix2D((x0,y0), theta, scale)  # 旋转变换矩阵
    imgRot1 = cv.warpAffine(img, MAR0, (width, height))

    # 以任意点为中心旋转
    x0,y0 = width/2, height/2 # 图像中心
    angle = theta * np.pi/180  #弧度->角度
    wRot = int(width*np.cos(angle) + height*np.sin(angle))  # 调整宽度
    hRot = int(height*np.cos(angle) + width*np.sin(angle))  # 调整高度
    scale = width/wRot
    MAR1 = cv.getRotationMatrix2D((x0,y0), theta, 1.0)
    MAR2 = cv.getRotationMatrix2D((x0,y0), theta, scale)

    imgRot2 = cv.warpAffine(img, MAR1, (height,width), borderValue=(255,255,255))
    imgRot3 = cv.warpAffine(img, MAR2, (height,width))
    print(img.shape, imgRot2.shape, imgRot3.shape, scale)

    # 图像直角旋转
    imgRot90 = cv.rotate(img,cv.ROTATE_90_CLOCKWISE)
    imgRot180 = cv.rotate(img,cv.ROTATE_180)
    imgRot270 = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Rotate around the origin"),plt.imshow(cv.cvtColor(imgRot1, cv.COLOR_BGR2RGB))
    plt.subplot(232),plt.axis('off'),plt.title("2. Rotate around the center"),plt.imshow(cv.cvtColor(imgRot2, cv.COLOR_BGR2RGB))
    plt.subplot(233),plt.axis('off'),plt.title("3. Rotate and resize"),plt.imshow(cv.cvtColor(imgRot3, cv.COLOR_BGR2RGB))

    plt.subplot(234),plt.axis('off'),plt.title("4. Rotate 90 degrees"),plt.imshow(cv.cvtColor(imgRot90, cv.COLOR_BGR2RGB))
    plt.subplot(235),plt.axis('off'),plt.title("5. Rotate 180 degrees"),plt.imshow(cv.cvtColor(imgRot180, cv.COLOR_BGR2RGB))
    plt.subplot(236),plt.axis('off'),plt.title("6. Rotate 270 degrees"),plt.imshow(cv.cvtColor(imgRot270, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


def ImageFlip(filepath1):
    img = cv.imread(filepath1)
    
    imgFlipV = cv.flip(img, 0)  # 垂直翻转
    imgFlipH = cv.flip(img, 1)  # 水平翻转
    imgFlipHV = cv.flip(img, -1)  # 水平垂直翻转

    plt.figure(figsize=(7,5))
    plt.subplot(221),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(222),plt.axis('off'),plt.title("2. Flip Horizontally 1"),plt.imshow(cv.cvtColor(imgFlipV, cv.COLOR_BGR2RGB))
    plt.subplot(223),plt.axis('off'),plt.title("3. Flip Vertically"),plt.imshow(cv.cvtColor(imgFlipH, cv.COLOR_BGR2RGB))
    plt.subplot(224),plt.axis('off'),plt.title("4. Flip Hori&Vert"),plt.imshow(cv.cvtColor(imgFlipHV, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def ImageShear(filepath1):
    img = cv.imread(filepath1)
    height, width = img.shape[:2]
    print("height,width = ",height, width)

    angle = 20 * np.pi/180  # 斜切角度

    # 水平斜切
    MAS = np.float32([[1, np.tan(angle), 0], [0, 1, 0]])  #斜切变换矩阵
    wShear = width + int(height * abs(np.tan(angle))) # 调整宽度
    imgShearH = cv.warpAffine(img, MAS, (wShear, height))

    # 垂直斜切
    MAS = np.float32([[1, 0, 0], [np.tan(angle), 1, 0]])  #斜切变换矩阵
    hShear = height + int(width * abs(np.tan(angle))) # 调整高度
    imgShear = cv.warpAffine(img, MAS, (width, wShear))

    plt.figure(figsize=(9,4))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. Horizontal shear"),plt.imshow(cv.cvtColor(imgShearH, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. Vertical shear"),plt.imshow(cv.cvtColor(imgShear, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def onMouseAction(event, x, y, flags, param):
    setpoint = (x, y)
    if event == cv.EVENT_LBUTTONDOWN: # 单击
        pts.append(setpoint)  # 选中一个点
        print("选择顶点{}: {}".format(len(pts), setpoint))

def ImageProjectionTransformation(filepath1):
    img = cv.imread(filepath1)
    imgCopy = img.copy()
    height, width = img.shape[:2]
    print("height,width = ",height, width)

    # 鼠标交互 从输入图像选择四个顶点
    print("单击左键选择4个顶点 (左上-左下-右下-右上) :")
    # pts = []    # 初始化 ROI顶点集合
    status = True   # 进入画图状态
    cv.namedWindow('origin')   # 创建显示窗口
    cv.setMouseCallback('origin', onMouseAction, status)

    while True :
        if len(pts) > 0:
            cv.circle(imgCopy, pts[-1], 5, (0,0,2255), -1)
        if len(pts) > 1:
            cv.line(imgCopy, pts[-1], pts[-2], (255,0,0), 2)
        if len(pts) == 4:
            cv.line(imgCopy, pts[0], pts[-1], (255,0,0), 2)
            cv.imshow('origin', imgCopy)
            cv.waitKey(1000)
            break
        cv.imshow('origin', imgCopy)
        cv.waitKey(100)
    cv.destroyAllWindows()
    ptsSrc = np.array(pts)
    print(ptsSrc)

    # 计算投影变换矩阵
    ptsSrc = np.float32(pts)  # 列表转换为Numpy数组，图像4个顶点坐标为(x,y)
    x1, y1, x2, y2 = int(0.1*width), int(0.1*height), int(0.9*width), int(0.9*height)
    ptsDst = np.float32([[x1,y1], [x1,y2], [x2,y2], [x2,y1]])  #投影变换后的4个顶点坐标
    MP = cv.getPerspectiveTransform(ptsSrc, ptsDst)

    # 投影变换
    dsize = (width, height)
    perspect = cv.warpPerspective(img, MP, dsize, borderValue=(255,255,255))
    print(img.shape, ptsSrc.shape, ptsDst.shape)

    plt.figure(figsize=(9,3.4))
    plt.subplot(131),plt.axis('off'),plt.title("1. img"),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. select vertex"),plt.imshow(cv.cvtColor(imgCopy, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. perspective correction"),plt.imshow(cv.cvtColor(perspect, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()





if __name__ == "__main__" :
    filepath1 = r"img/lenna.bmp"
    filepath2 = r"img/remap.png"

    
    # ImageTranslation(filepath1)
    # ImageResize(filepath1)
    # ImageRotate(filepath1)
    # ImageFlip(filepath1)
    # ImageShear(filepath1)
    
    pts = []  # # 初始化 ROI顶点集合
    # ImageProjectionTransformation(filepath2)

