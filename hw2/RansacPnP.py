from scipy.spatial.transform import Rotation as Rot
import pandas as pd
import numpy as np
import cv2
import time

from IPython.display import display

from p3p import P3P

def reproject_error(points2D, points3D, rvec, tvec, cameraMatrix, distCoeffs):
    total_error = 0

    project_points,_ = cv2.projectPoints(points3D, rvec, tvec, cameraMatrix, distCoeffs)
    project_points = project_points.reshape(-1,2)

    for i, project_point in enumerate(project_points):
        pixel_point = points2D[i]

        total_error += np.linalg.norm(pixel_point - project_point)

    return total_error

def RANSACPnP(points3D, points2D, cameraMatrix, distCoeffs, times = 100):
    # Finding the least reprojection error
    best_error = np.inf
    best_T = None
    best_R = None

    for i in range(times):
        # try:
        random_sample = np.random.choice(points2D.shape[0], 4, replace=False)
        P3P_c = P3P(cameraMatrix, distCoeffs)
        output = P3P_c.solve_P4P(points2D[random_sample], points3D[random_sample])

        if len(output) == 0:
            continue
        else:
            rvec, tvec = output

        error = reproject_error(points2D, points3D, rvec, tvec, cameraMatrix, distCoeffs)
        
        if error < best_error:
            best_error = error
            best_T = tvec
            best_R = rvec

    print("best_error: ", best_error)

    return best_R, best_T