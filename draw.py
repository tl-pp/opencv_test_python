import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def drawLine():
    h,w,ch = 180,200,3
    img = np.ones((h,w,ch),np.uint8)*160 # 创建灰色图像

    # 1.线条参数color设置
    img1 = img.copy()
    cv.line(img1, (0,0),(200,180), (0,0,255), 1)  # 红
    cv.line(img1, (0,0),(100,180), (0,255,0), 1)  # 绿
    cv.line(img1, (0,40),(200,40), (128,0,0), 2)  # 蓝
    cv.line(img1, (0,80),(200,80), 128, 2) # 等效 （128，0，0）
    cv.line(img1, (0,120),(200,120), 255, 2) # 等效 255

    # 2.线宽的设置
    img2 = img.copy()
    cv.line(img2, (20,50), (180,10), (255,0,0), 1, cv.LINE_8)
    cv.line(img2, (20,90), (180,50), (255,0,0), 1, cv.LINE_AA)

    print("cv.LINE_8 = ", cv.LINE_8)
    print("cv.LINE_AA = ", cv.LINE_AA)
    cv.line(img2, (20,130), (180,90), (255,0,0), cv.LINE_8)  # cv.LINE_8被认为是线宽
    cv.line(img2, (20,170), (180,130), (255,0,0), cv.LINE_AA) # cv.LINE_AA被认为是线宽

    # 3. tiplength指箭头部分长度与整个线段长度的比例
    img3 = img.copy()
    img3 = cv.arrowedLine(img3, (20,20), (180,20), (0,0,255), tipLength=0.05)
    img3 = cv.arrowedLine(img3, (20,60), (180,60), (0,0,255), tipLength=0.05)

    img3 = cv.arrowedLine(img3, (20,100), (180,100), (0,0,255), tipLength=0.15)  # 双向
    img3 = cv.arrowedLine(img3, (180,100), (20,100), (0,0,255), tipLength=0.15)

    img3 = cv.arrowedLine(img3, (20,140), (210,140), (0,0,255), tipLength=0.2) # 终点越界，箭头未显示

    # 4.没有复制原图像，直接输入img，会互相影响
    img4 = cv.line(img, (0,100), (150,100), (0,255,0),1)  # 水平线
    img5 = cv.line(img, (75,0), (75,200), (0,255,0), 1)   # 垂直线

    # 5.灰度图像只能绘制灰度线条
    img6 = np.zeros((h,w), np.uint8) 
    cv.line(img6, (0,10), (200,10), (0,255,255), 2)  # Gray = 0
    cv.line(img6, (0,30), (200,30), (64,128,255), 2)  # Gray = 64
    cv.line(img6, (0,60), (200,60), (128,64,255), 2)  # Gray = 128
    cv.line(img6, (0,100), (200,100), (255,0,255), 2)  # Gray = 255
    cv.line(img6, (20,0), (20,200), 128, 2)  # Gray = 128
    cv.line(img6, (60,0), (60,200), (255,0,0), 2)  # Gray = 255
    cv.line(img6, (100,0), (100,200), (255,255,255), 2)  # Gray = 255

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. img1"),plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(232),plt.axis('off'),plt.title("2. img2"),plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(233),plt.axis('off'),plt.title("3. img3"),plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.subplot(234),plt.axis('off'),plt.title("4. img4"),plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
    plt.subplot(235),plt.axis('off'),plt.title("5. img5"),plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
    plt.subplot(236),plt.axis('off'),plt.title("6. img6"),plt.imshow(img6, cmap="gray")
    plt.tight_layout()
    plt.show()


def drawRectangle():
    height,width,ch = 300,320,3
    img = np.ones((height,width,ch),np.uint8) * 192

    # 1.矩形参数设置为p1(x1,y1) p2(x2,y2)
    img1 = img.copy()
    cv.rectangle(img1, (0,80), (100,220), (0,0,255), 2)
    cv.rectangle(img1, (80,0), (220,100), (0,255,0), 2)
    cv.rectangle(img1, (150,120), (400,200), 255, 2) #越界自动裁剪
    cv.rectangle(img1, (50,10), (100,50), (128,0,0), 1) #线宽
    cv.rectangle(img1, (150,10), (200,50), (192,0,0), 2)
    cv.rectangle(img1, (250,10), (300,50), (255,0,0), 4)
    cv.rectangle(img1, (50,250), (100,290), (128,0,0), -1) #内部填充
    cv.rectangle(img1, (150,250), (200,290), (192,0,0), -1)
    cv.rectangle(img1, (250,250), (300,290), (255,0,0), -1)

    # 2.通过 x,y,w,h 绘制矩形（c++ 则是rect类）
    img2 = img.copy()
    x,y,w,h = (50,100,200,100)
    cv.rectangle(img2, (x,y), (x+w,y+h), (0,0,255), 2)
    text = "({},{}),{}*{}".format(x,y,w,h)
    cv.putText(img2, text, (x,y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

    # 3.在灰度图像中绘制主线和矩形
    img3 = np.zeros((height,width), np.uint8)
    cv.line(img3, (0,40), (320,40), 64, 2)
    cv.line(img3, (0,80), (320,80), 128, 2)
    cv.line(img3, (0,120), (320,120), 192, 2)

    cv.rectangle(img3, (20,250), (50,220), 128, -1)
    cv.rectangle(img3, (80,250), (110,210), 128, -1)
    cv.rectangle(img3, (140,250), (170,200), 128, -1)
    cv.rectangle(img3, (200,250), (230,190), 192, -1)
    cv.rectangle(img3, (260,250), (290,180), 255, -1)

    plt.figure(figsize=(9,3.3))
    plt.subplot(131),plt.axis('off'),plt.title("1. img1"),plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. img2"),plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(133),plt.axis('off'),plt.title("3. img3"),plt.imshow(img3, cmap="gray")
    plt.tight_layout()
    plt.show()


def drawRotateRect():
    height,width,channels = 300,400,3
    img = np.ones((height,width,channels), np.uint8) * 192

    # 1.围绕矩形中间旋转
    cx,cy,w,h = (200,150,200,100) #左上角坐标（x,y）,宽度w和高度h
    img1 = img.copy()
    cv.circle(img1, (cx,cy), 4, (0,0,255), -1) #旋转中心
    angle = [15,30,45,60,75,90] #旋转角度，顺时针
    box = np.zeros((4,2), np.int32) #计算旋转矩形顶点（4，2）
    for i in range(len(angle)):
        rect = ((cx,cy), (w,h), angle[i])
        box = np.int32(cv.boxPoints(rect))
        color = (30*i, 0, 255 - 30*i)
        cv.drawContours(img1, [box], 0, color, 1)
        print(rect)
    
    # 2.围绕左上顶点旋转
    x,y,w,h = (200,100,160,100) #左上角坐标（x,y）,宽度w和高度h
    img2 = img.copy()
    cv.circle(img2, (x,y), 4, (0,0,255), -1) #旋转中心
    angle = [15,30,45,60,75,90,120,150,180,225] #旋转角度，顺时针
    for i in range(len(angle)):
        ang = angle[i] * np.pi/180
        x1, y1 = x,y
        x2 = int(x + w*np.cos(ang))
        y2 = int(y + w*np.sin(ang))
        x3 = int(x + w*np.cos(ang) - h*np.sin(ang))
        y3 = int(y + w*np.sin(ang) + h*np.cos(ang))
        x4 = int(x - h*np.sin(ang))
        y4 = int(y + h*np.cos(ang))
        color = (30*i,0,255-30*i)
        box = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        cv.drawContours(img2, [box], 0, color, 1)

    plt.figure(figsize=(9,3.2))
    plt.subplot(131),plt.axis('off'),plt.title("1. img1"),plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(132),plt.axis('off'),plt.title("2. img2"),plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

def drawCircle():
    img = np.ones((400,600,3), np.uint8) * 192

    center = (0,0)
    cx, cy = 300,200
    for r in range(200,0,-20):
        color = (r,r,255-r)
        cv.circle(img, (cx,cy),r,color, -1)
        cv.circle(img, center, r, 255)
        cv.circle(img, (600,400), r, color, 5)
    plt.figure(figsize=(6,4))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def drawPolygon():
    img = np.ones((900,400,3), np.uint8) * 224
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()

    # 多边形顶点
    points1 = np.array([[200,60], [295,129], [259,241], [141,241], [105,129]], np.int32)
    points2 = np.array([[200,350], [259,531], [105,419], [295,419], [141,531]], np.int32)
    points3 = np.array([[200,640], [222,709], [295,709], [236,752], [259,821], [200,778], [141,821], [164,752], [105,709], [178,709]], np.int32)

    print(points1.shape, points2.shape, points3.shape)

    # 多边形闭合曲线
    pts1 = [points1]
    cv.polylines(img1, pts1, True, (0,0,255)) # pts1 是列表
    cv.polylines(img1, [points2, points3], True, 255, 2) # 也可以绘制多个

    # 多边形不闭合曲线
    cv.polylines(img2, pts1, False, (0,0,255))
    cv.polylines(img2, [points2, points3], False, 255, 2)

    # 填充多边形
    cv.fillPoly(img3, [points1], (0,0,255))
    cv.fillPoly(img3, [points2, points3], 255)

    # 填充多边形，注意交叉重叠部分
    cv.fillConvexPoly(img4, points1, (0,0,255))
    cv.fillConvexPoly(img4, points2, 255)
    cv.fillConvexPoly(img4, points3, 255)

    plt.figure(figsize=(9,5.3))
    plt.subplot(141),plt.axis('off'),plt.title("1. closed polygon"),plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(142),plt.axis('off'),plt.title("2. unclosed polygon"),plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(143),plt.axis('off'),plt.title("3. fillPoly"),plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.subplot(144),plt.axis('off'),plt.title("4. fillConvexPoly"),plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # drawLine()
    # drawRectangle()
    # drawRotateRect()
    # drawCircle()
    drawPolygon()
