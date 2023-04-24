import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as Rot

def draw_points(rimg, transform_matrix, position, all_points, colors):
    position = np.array(position)

    # sorting points from the one nearest to camera to furthest
    for i, points in enumerate(all_points):

        for point in points:
            point = np.append(point,1)
            pos = transform_matrix@np.array(point)
            pos /= pos[2]
            x,y = int(pos[0]), int(pos[1])
            cv2.circle(rimg, (x,y), 5, colors[i], -1)

    return rimg

cube_vertices = np.load("cube_vertices.npy")

TOP = [0,1,2]
BOTTOM = [6,4,7]
LEFT = [0,2,4]
RIGHT = [7,3,5]
FRONT = [2,3,6]
BACK = [4,0,5]
sides = [TOP, BOTTOM, LEFT, RIGHT, FRONT, BACK]

# position of all the points of the 6 sides
all_points  = []

for i, side in enumerate(sides):
    pos = cube_vertices[side]
    origin = pos[0]
    width = pos[1]-pos[0]
    height = pos[2]-pos[0]

    # In 1 side, there will be (amt x amt) number of dots
    amt = 10
    points = []
    for x in range(amt):
        for y in range(amt):
            point = origin + x/amt * width + y/amt * height
            points.append(np.array(point))
    
    points = np.array(points)
    all_points.append(points)

all_points = np.array(all_points)

images_df = pd.read_pickle("data/images.pkl")
position_df = pd.read_pickle("camera_position.pkl")

# 1 color for each side
colors = [(0,0,255), (0,255,0), (0,255,255), (255,0,0), (255,0,255), (255,255,0)]

for i in trange(293):
    # Load quaery image 
    idx = i+1
    fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
    rimg = cv2.imread("data/frames/"+fname)

    # Load camera positions
    camera_position = position_df.iloc[i]['position']

    # Get camera pose groudtruth 
    ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values

    # Get rotation and translation
    R = Rot.from_quat(rotq_gt).as_matrix().reshape(3,3)
    T = tvec_gt.reshape(3,1)

    # Get camera's extrinsic matrix
    RT = np.concatenate((R,T), axis = 1)

    # Get camera's intrinsic matrix
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])

    #Get camera's matrix
    P = cameraMatrix@RT

    cube_img = draw_points(rimg, P, camera_position, all_points, colors)
    cv2.imwrite("cube_images/"+str(idx)+".jpg", cube_img)