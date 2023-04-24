import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

def find_homography(points1,points2):
    matrices = []

    for i in range(points1.shape[0]):
        x1,y1 = points1[i]
        x2,y2 = points2[i]

        matrices.append([x1, y1, 1, 0, 0, 0, -x1*x2, -x2*y1, -x2])
        matrices.append([0, 0, 0, x1, y1, 1, -x1*y2, -y2*y1, -y2])

    A = np.array(matrices)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(matrices)

    H = np.reshape(V[-1], (3,3))
    H = (1 / H.item(8)) * H

    return H

def bilinear_interpolation(img, x, y):
    x1 = math.floor(x)
    x2 = math.floor(x+1)
    y1 = math.floor(y)
    y2 = math.floor(y+1)

    x_d1 = x-float(x1)
    x_d2 = float(x2)-x
    y_d1 = y-float(y1)
    y_d2 = float(y2)-y

    img=np.array(img)

    wa = x_d2*y_d2
    wb = x_d1*y_d2
    wc = x_d1*y_d1
    wd = x_d2*y_d1

    a = img[x1][y1].astype(float)
    b = img[x2][y1].astype(float)
    c = img[x2][y2].astype(float)
    d = img[x1][y2].astype(float)

    result = 0
    result += a * wa
    result += b * wb
    result += c * wc
    result += d * wd

    result.astype(int)
    return result

def transform(img, H, shape):

    new_img = np.zeros((shape[0], shape[1], 3), dtype='uint8')
    
    # Finding the inverse of homography for backward warping
    inv_H = np.linalg.inv(H)
    inv_H = inv_H/inv_H[2][2]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Backward Warping, finding the position in the original image
            mul = np.matmul(inv_H, [j,i,1])
            mul = mul/mul[-1] # dividing by z

            y = mul[0]
            x = mul[1]

            try:
                # Find the correct color intensity through bilinear interpolation
                new_img[i][j] = bilinear_interpolation(img, x, y)
            except:
                continue

    return new_img



if __name__ == '__main__':
    # img3 = cv.imread('sys.argv[1]')
    img3 = cv.imread('images/2.png')
    shape = img3.shape

    # Q2-1
    # manually added points: [[297, 191], [574, 410], [305, 743], [28, 516]]
    points1 = np.array([[297, 191], [574, 410], [305, 743], [28, 516]])
    points2 =  np.array([[0, 0], [shape[1]-1, 0], [shape[1]-1, shape[0]-1], [0, shape[0]-1]])

    H = find_homography(points1, points2)

    # Q2-2
    new_img = transform(img3, H, shape)
    cv.imwrite('images/2_warp.png', new_img)