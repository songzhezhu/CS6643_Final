'''
Name: HoughTransform.py
Purpose: This is the final project for CS6643 Computer Vision course.
Author: Songzhe Zhu, Xuelin Yu, Yiming Zhang
'''
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np


def drawPic(name, imgArray):
    cv2.imwrite(name, imgArray)
    cmd = "open " + name
    os.system(cmd)
    return 0

def convert_to_gray(image):
    # Convert source image to gray image
    size = image.shape
    height = size[0]
    width = size[1]
    gray_img = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            gray_img[i, j] = (0.3 * image[i, j, 2]) + (0.59 * image[i, j, 1]) + (0.11 * image[i, j, 0])
    return gray_img


def Gaussian_filtering(image):
    # Guassian_filtering
    smooth_img = image.copy()
    height, width = image.shape
    gaussian = np.array([0.03, 0.07, 0.12, 0.18, 0.20, 0.18, 0.12, 0.07, 0.03])
    for i in range(height):
        for j in range(4, width - 4):
            smooth_img[i, j] = np.sum(image[i, j - 4: j + 5] * gaussian)
    for j in range(width):
        for i in range(4, height - 4):
            smooth_img[i, j] = np.sum(smooth_img[i - 4: i + 5, j] * gaussian)
    return smooth_img


def Canny(image):
    # Implement Canny edge detector
    # Step 1: convert to gray image and use Gaussian filter as Optimal Operator
    gray_img = convert_to_gray(image)
    smooth_img = Gaussian_filtering(gray_img)

    height, width = smooth_img.shape
    # Step 2: compute gradient magnitude
    dx = np.zeros([height - 1, width - 1])
    dy = np.zeros([height - 1, width - 1])
    delta = np.zeros([height - 1, width - 1])
    for i in range(height - 1):
        for j in range(width - 1):
            dx[i, j] = smooth_img[i, j + 1] - smooth_img[i, j]
            dy[i, j] = smooth_img[i + 1, j] - smooth_img[i, j]
            delta[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))

    # Setp 3: Non-Maximum-Suppression
    dwidth, dheight = delta.shape
    nms = np.copy(delta)
    nms[0, :] = nms[dwidth - 1, :] = nms[:, 0] = nms[:, dheight - 1] = 0
    for i in range(1, dwidth - 1):
        for j in range(1, dheight - 1):
            if delta[i, j] == 0:
                nms[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                tempgrad = delta[i, j]
                # if gradient in y-axis direction is larger
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = delta[i - 1, j]
                    grad4 = delta[i + 1, j]
                    # if directions in x and y are the same
                    if gradX * gradY > 0:
                        grad1 = delta[i - 1, j - 1]
                        grad3 = delta[i + 1, j + 1]
                    # if directions in x and y are opposite
                    else:
                        grad1 = delta[i - 1, j + 1]
                        grad3 = delta[i + 1, j - 1]

                # if gradient in x-axis direction is larger
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = delta[i, j - 1]
                    grad4 = delta[i, j + 1]
                    # if directions in x and y are the same
                    if gradX * gradY > 0:
                        grad1 = delta[i + 1, j - 1]
                        grad3 = delta[i - 1, j + 1]
                    # if directions in x and y are opposite
                    else:
                        grad1 = delta[i - 1, j - 1]
                        grad3 = delta[i + 1, j + 1]

                gradtemp1 = weight * grad1 + (1 - weight) * grad2
                gradtemp2 = weight * grad3 + (1 - weight) * grad4
                if tempgrad >= gradtemp1 and tempgrad >= gradtemp2:
                    nms[i, j] = tempgrad
                else:
                    nms[i, j] = 0


# step4. 双阈值算法检测、连接边缘
    nmsw, nmsh = nms.shape
    res = np.zeros([nmsw, nmsh], np.uint8)
    # 定义高低阈值
    tl = 0.1 * np.max(nms)
    th = 0.3 * np.max(nms)
    for i in range(1, nmsw - 1):
        for j in range(1, nmsh - 1):
            if nms[i, j] < tl:
                res[i, j] = 0
            elif nms[i, j] > th:
                res[i, j] = 255
            elif any(nms[i - 1, j - 1:j + 1] < th) or any(nms[i + 1, j - 1:j + 1] < th) or any(nms[i, j - 1:j + 1] < th):
                res[i, j] = 255
    return res


# Counter-clockwise rotate the image
def rotateImage(src, degree):
    h, w = src.shape[:2]
    # Calculate the affine transformation matrix
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # Affine transformation
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate


def houghLines(img, threshold):
    rhoMax = (img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5
    rhoIndexRange = int(round(2 * rhoMax)) + 1
    tableThetaRho = np.zeros((rhoIndexRange, 181), dtype=np.int32);
    res = ([0])
    n = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (img[y, x] == 255):
                for r in range(0, 181):
                    theta = float(float(r) / 180 * np.pi)
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    rhoIndex = int(round(rho + rhoMax))
                    tableThetaRho[rhoIndex, r] += 1
    for i in range(181):
        for j in range(rhoIndexRange):
            if (tableThetaRho[j, i] >= threshold):
                theta = float(float(i) / 180 * np.pi)
                rho = j - rhoMax
                res.append(rho)
                res.append(theta)
                n += 1
                #print (">threshold", theta, i, rho)
    res = np.array(res[1:n * 2 + 1]).reshape(n, 2)
    return res


# Use hough transform to calculate the rotate angle
def CalcDegree(srcImage, th, clock):
    dstImage = Canny(srcImage)
    drawPic('canny.png', dstImage)
    lineimage = srcImage.copy()

    # Hough transform
    #TODO: 自己实现霍夫变换
    #lines = cv2.HoughLines(dstImage, 1, np.pi / 180, th)
    # Difference depends on threshold.
    # High threshold makes hard to detect lines，
    # Low threshold slows speed down.
    lines = houghLines(dstImage, th)

    sum = 0
    numlines = 0

    # draw lines which hough transform calculated
    for i in range(len(lines)):
        rho = lines[i,0]
        theta = lines[i,1]
        if theta == 0.0:
            continue
        # print("theta:", theta, " rho:", rho)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(round(x0 + 1000 * (-b)))
        y1 = int(round(y0 + 1000 * a))
        x2 = int(round(x0 - 1000 * (-b)))
        y2 = int(round(y0 - 1000 * a))
        sum += theta
        numlines += 1
        cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    # Take the average for the theta values
    average = sum / numlines
    res = average / np.pi * 180
    angle = res - 90
    if clock == 0:
        angle += 180
    drawPic('edge.png', lineimage)
    return angle, lineimage, dstImage


if __name__ == '__main__':
    gold = 'y'
    while 1:
        if gold == 'y':
            input_img_file = input("----Type the file name you want use: ")
            image = cv2.imread(input_img_file)
            if image is None:
                print("!!!! This file is not exist, try again")
                continue
            clock = int(input("----You want clockwise rotation or counter-clockwise?: type '0' for clockwise, '1' for counter-clockwise: "))
            threshold = input("----Choose a threshold for the Hough Transform: ")
            if threshold.isdigit():
                threshold = int(threshold)
            else:
                threshold = int(input("!!!! Enter a digit number please: "))

        degree, img_edge, canny = CalcDegree(image, threshold, clock)
        rotate = rotateImage(image, degree)
        print(">>>After calculation, here is the expecting rotate angle based on your threshold: ", degree)
        drawPic("rotate.png", rotate)
        gold = input("----Feel good about the result? If so, type 'y', otherwise, type 'n': ")
        if gold == 'n':
            threshold = int(input("----Choose a new threshold: "))
            clock = int(input(
                "----You want clockwise rotation or counter-clockwise?: type '0' for clockwise, '1' for counter-clockwise: "))
            image = image
        elif gold == 'y':
            cont = input("----Want try another image? If so, type 'y'; Quit? type 'q': ")
            if cont == 'y':
                continue
            else:
                break

