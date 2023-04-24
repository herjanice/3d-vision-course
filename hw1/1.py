from locale import normalize
import sys
import numpy as np
import cv2 as cv
from tqdm import tqdm, trange
import random

def get_sift_correspondences(img1, img2, k=0, useRANSAC = False):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''

    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # Applying ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # selecting top k features
    if k!=0 and useRANSAC == False:
        good_matches = good_matches[0:k]
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    if useRANSAC:
        new_points1, new_points2, combinations = RANSAC(points1, points2, trials=10000, k=k)
        points1 = new_points1
        points2 = new_points2

    good_matches = np.array(good_matches)
    
    # img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches[combinations], None, flags=cv.DrawMatchesFlags_DEFAULT)
    # cv.imshow('match', img_draw_match)
    # cv.waitKey(0)
    # cv.imwrite('1_ransac_k'+str(k)+'.jpg', img_draw_match)

    return points1, points2

def experiment(img1, img2, k=0, method = 'ORB'):

    if method == 'ORB':
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

        img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
        search_params = dict(checks=100)

        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)


        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # selecting top k features
        if k!=0:
            good_matches = good_matches[0:k]
        points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

        # img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv.imwrite('orb_flann_1_k'+str(k)+'.jpg', img_draw_match)

    return points1, points2

def RANSAC(points1, points2, trials, k):
    # tries out different combinations of matches to compute the best homography matrix
    points1 = np.array(points1)
    points2 = np.array(points2)

    best_inliers = []

    best_error = 10000

    for trial in trange(trials):

        # randomly sample from the correspondence
        rand = np.random.default_rng()
        randomSample = rand.choice(len(points1), size=k, replace=False)
        
        chosen_points1 = points1[randomSample]
        chosen_points2 = points2[randomSample]

        H = find_homography(chosen_points1, chosen_points2)
        projection_err = calculate_projection_error(points1, points2, H)

        if projection_err < best_error:
            best_error = projection_err
            best_inliers = randomSample

    return points1[best_inliers], points2[best_inliers], best_inliers


def find_homography(points1,points2):
    matrices = []

    for i in range(points1.shape[0]):
        x1,y1 = points1[i]
        x2,y2 = points2[i]

        matrices.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
        matrices.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])

    A = np.array(matrices)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(matrices)

    H = np.reshape(V[-1], (3,3))
    H = (1 / H.item(8)) * H

    return H

def calculate_projection_error(p_source, p_target, H):
    error = 0
    
    for i in range(p_source.shape[0]):
        mul = np.matmul(H, [p_source[i][0], p_source[i][1], 1])
        mul /= mul[-1] # divide by z to get x and y

        x = int(mul[0])
        y = int(mul[1])

        error += (((p_target[i][0]-x)**2 + (p_target[i][1]-y)**2)**0.5)

    error /= p_source.shape[0]

    return error

def normalize_points(points):
    mean_0 = np.mean(points[:,0])
    mean_1 = np.mean(points[:,1])

    scale_rate = np.mean(np.sqrt(np.sum((points[:]-[mean_0, mean_1])**2,axis=1)))

    shift = np.array([[-1, 0, -1*mean_0], [0, 1, -1*mean_1], [0, 0, 1]])
    scale = np.array([[1/scale_rate, 0, 0],[0, 1/scale_rate, 0],[0, 0, 1]])
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

    matrix = np.matmul(scale, shift)
    new_points = np.matmul(matrix, points.T).T[:, 0:2]

    return matrix, new_points

def find_normalized_homography(points1, points2):

    matrix1, new_points1 = normalize_points(points1)
    matrix2, new_points2 = normalize_points(points2)

    H = find_homography(new_points1, new_points2)
    NH = np.matmul(np.linalg.inv(matrix2), np.matmul(H,matrix1))
    NH /= NH[2][2]

    return NH
    

if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])

    p_source = gt_correspondences[0]
    p_target = gt_correspondences[1]

    
    # 1-1
    k=4
    points1, points2 = get_sift_correspondences(img1, img2, k, useRANSAC = True)
    H = find_homography(points1, points2)

    # 1-2
    projection_err = calculate_projection_error(p_source, p_target, H)

    # 1-3
    NH = find_normalized_homography(points1, points2)
    norm_projection_error = calculate_projection_error(p_source, p_target, NH)



    print(H)
    print("DLT projection error = ", projection_err)
    print(NH)
    print("Normalized DLT projection error = ", norm_projection_error)

    
    
