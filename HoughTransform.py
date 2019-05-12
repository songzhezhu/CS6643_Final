'''
Name: HoughTransform.py
Purpose: This is the final project for CS6643 Computer Vision course.
Author: Songzhe Zhu, Xuelin Yu, Yiming Zhang
'''

import cv2
import numpy as np
import math

input_img_file = "resize2.png"


def canny(img):
    return 0


# 度数转换
def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res


# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    print(RotateMatrix)
    # 仿射变换，背景色填充为黑色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(0, 0, 0))
    return rotate

def switch_grayscale(image):
    gray_value = 0.3 * image[:, :, 2] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 0]
    gray_img = gray_value.astype(np.uint8)
    return gray_img

# 通过霍夫变换计算角度
def CalcDegree(srcImage):
    midImage = switch_grayscale(srcImage)
    #TODO：自己实现canny算子
    dstImage = cv2.Canny(midImage, 50, 200, 3)
    lineimage = srcImage.copy()

    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    #TODO: 自己实现霍夫变换
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 200)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    sum = 0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # 只选角度最小的作为旋转角度
            sum += theta
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    # 对所有角度求平均，这样做旋转效果会更好
    average = sum / len(lines)
    angle = DegreeTrans(average) - 90
    return angle, lineimage, dstImage


if __name__ == '__main__':
    image = cv2.imread(input_img_file)
    # 倾斜角度矫正
    degree, img_edge, canny = CalcDegree(image)
    print("调整角度：", degree)
    rotate = rotateImage(image, degree)
    cv2.imwrite("rotate.png", rotate)
    cv2.imwrite('edge.png', img_edge)
    cv2.imwrite('canny.png', canny)
