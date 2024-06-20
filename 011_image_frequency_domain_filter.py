import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# 傅里叶变换与频域滤波 

def ImageOpencv2DFFT(filepath):
    img = cv.imread(filepath, flags=0)

    # 图像的傅里叶变换
    imgFloat = img.astype(np.float32)    # 将图像转成float32
    dft = cv.dft(imgFloat, flags=cv.DFT_COMPLEX_OUTPUT)  # (512, 512, 2)
    dftShift = np.fft.fftshift(dft)      # 将低频分量移到频谱中心

    # 图像的傅里叶逆变换
    iShift = np.fft.ifftshift(dftShift)  # 将低频分量移回到四角
    idft = cv.idft(iShift)                  # (512, 512, 2)
    idftAmp = cv.magnitude(idft[:, :, 0], idft[:, :, 1]) # 重建图像
    rebuild = np.uint8(cv.normalize(idftAmp, None, 0, 255, cv.NORM_MINMAX))
    
    print("img: {}, dft:{}, idft:{}".format(img.shape, dft.shape, idft.shape))

    # 傅里叶频谱的显示
    dftAmp = cv.magnitude(dft[:, :, 0], dft[:, :, 1])    # 幅度谱，未中心化
    ampLog = np.log(1 + dftAmp)                          # 幅度谱对数变换，以便显示
    shiftDftAmp = cv.magnitude(dftShift[:, :, 0], dftShift[:, :, 1])  # 幅度谱中心化
    shiftAmplog = np.log(1 + shiftDftAmp)                # 幅度谱中心化对数变换，以便显示
    phase = np.arctan2(dft[:, :, 0], dft[:, :, 1])       # 相位谱 (弧度制)
    dftPhi = phase / np.pi * 180                         # 转换为角度制 [-180, 180]

    print("img min/max: {}, {}".format(imgFloat.min(), imgFloat.max()))
    print("dftMag min/max: {:.1f}, {:.1f}".format(dftAmp.min(), dftAmp.max()))
    print("dftPhi min/max: {:.1f}, {:.1f}".format(dftPhi.min(), dftPhi.max()))
    print("ampLog min/max: {:.1f}, {:.1f}".format(ampLog.min(), ampLog.max()))
    print("rebuild min/max: {}, {}".format(rebuild.min(), rebuild.max()))

    plt.figure(figsize=(9,6))
    plt.subplot(231),plt.axis('off'),plt.title("1. Original"),plt.imshow(img, cmap='gray')
    plt.subplot(232),plt.axis('off'),plt.title("2. DFT Phase"),plt.imshow(dftPhi, cmap='gray')
    plt.subplot(233),plt.axis('off'),plt.title("3. DFT amplitude"),plt.imshow(dftAmp, cmap='gray')
    plt.subplot(234),plt.axis('off'),plt.title("4. LogTrans of amplitude"),plt.imshow(ampLog, cmap='gray')
    plt.subplot(235),plt.axis('off'),plt.title("5. Shift to center"),plt.imshow(shiftAmplog, cmap='gray')
    plt.subplot(236),plt.axis('off'),plt.title("6. rebuild image whit IDFT"),plt.imshow(rebuild, cmap='gray')
    plt.tight_layout()
    plt.show()





    


if __name__ == "__main__":
    filepath1 = r"img/Fig1101.png"

    # opencv二维离散傅里叶变换
    ImageOpencv2DFFT(filepath1)