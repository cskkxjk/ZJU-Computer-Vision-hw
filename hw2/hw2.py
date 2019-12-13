#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# sobel
sobel_1 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])


def imgConvolve(image, kernel):
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)
    convolve_h = int(img_h + 2 * padding_h)
    convolve_w = int(img_w + 2 * padding_w)

    img_padding = np.zeros((convolve_h, convolve_w))
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    image_convolve = np.zeros(image.shape)
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h+1, j - padding_w:j + padding_w+1]*kernel))

    return image_convolve


def harris_calR(img, window_size=2, k=0.04):
    H, W = img.shape
    window = np.ones((window_size, window_size))
    R = np.zeros((H, W))
    lambda_1 = np.zeros((H, W))
    lambda_2 = np.zeros((H, W))
    padding_h = window_size // 2
    padding_w = window_size // 2
    sobel_x = imgConvolve(gray, sobel_1)
    sobel_y = imgConvolve(gray, sobel_2)
    sobel_x_padding = np.zeros((H + 2 * padding_h, W + 2 * padding_w))
    sobel_y_padding = np.zeros((H + 2 * padding_h, W + 2 * padding_w))
    sobel_x_padding[padding_h:padding_h + H, padding_w:padding_w + W] = sobel_x[:, :]
    sobel_y_padding[padding_h:padding_h + H, padding_w:padding_w + W] = sobel_y[:, :]
    Ix2 = sobel_x_padding ** 2
    Iy2 = sobel_y_padding ** 2
    Ixy = sobel_x_padding * sobel_y_padding
    M = np.zeros((2, 2))
    for i in range(H):
        for j in range(W):
            M[0, 0] = np.sum(Ix2[i:i + window_size, j:j + window_size] * window)
            M[0, 1] = np.sum(Ixy[i:i + window_size, j:j + window_size] * window)
            M[1, 0] = M[0, 1]
            M[1, 1] = np.sum(Iy2[i:i + window_size, j:j + window_size] * window)
            [l_1, l_2] = np.linalg.eigvals(M)
            if l_1 > l_2:
                lambda_1[i, j] = l_2
                lambda_2[i, j] = l_1
            else:
                lambda_1[i, j] = l_1
                lambda_2[i, j] = l_2
            R[i, j] = np.linalg.det(M) - k * (np.trace(M) ** 2)

    return R, lambda_1, lambda_2


def paint_R(R, imgshape):
    H, W = R.shape
    response = np.zeros(imgshape)
    min = R.min()
    max = R.max()
    scale = R.max() - R.min()
    color_part = scale / 3
    gap = color_part / 255
    blue_ceil = R.min() + color_part
    green_ceil = blue_ceil + color_part
    response[R < R.max()] = [0, 255, 0]
    response[R < 0.0001 * R.max()] = [255, 0, 0]
    response[R > 0.01 * R.max()] = [0, 0, 255]
    return response


img1 = cv.imread('test1.jpg')
img2 = cv.imread('test2.jpg')
h, w, d = img1.shape
image = np.concatenate((img1, img2), axis=1)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
black1 = np.zeros(image.shape)
black2 = np.zeros(image.shape)
black3 = np.zeros(image.shape)
r, lambda1, lambda2 = harris_calR(gray, window_size=3, k=0.04)
black1[lambda1 > 0.01 * lambda1.max()] = [255, 255, 255]
black2[lambda2 > 0.01 * lambda2.max()] = [255, 255, 255]
black3[r > 0.01 * r.max()] = [255, 255, 255]
cv.imshow('lambda1', black1)
cv.imwrite('min_lambda.jpg', black1)
cv.imshow('lambda2', black2)
cv.imwrite('max_lambda.jpg', black2)
cv.imshow('R', black3)
cv.imwrite('R.jpg', black3)
colored_R = paint_R(r, image.shape)
cv.imshow('coloredR', colored_R)
cv.imwrite('coloredR.jpg', colored_R)
image[r > 0.01 * r.max()] = [0, 0, 255]
cv.imshow('dst', image)
cv.imwrite('dst.jpg', image)
if cv.waitKey(0) & 0xff == '27':
    cv.destroyAllWindows()
