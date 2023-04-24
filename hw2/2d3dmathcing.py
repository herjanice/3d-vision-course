from scipy.spatial.transform import Rotation as Rot
import pandas as pd
import numpy as np
import cv2
import time

from IPython.display import display
from tqdm import tqdm, trange

from RansacPnP import RANSACPnP

images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")
print("data loading success...")

# display(images_df)
# display(train_df)
# display(points3D_df)
# display(point_desc_df)

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    #compute the average per row
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    # display(desc)
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solvePnPRansac
    # return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
    return RANSACPnP(points3D, points2D, cameraMatrix, distCoeffs, times=100)

def find_camera_position(rvec, tvec):
    R=Rot.from_rotvec(rvec.reshape(1,3)).as_matrix().reshape(3,3)
    T = tvec.reshape(3,1)

    # camera's extrinsic matrix
    R_T=np.concatenate((R, T), axis=1)
    tmp=np.array([[0,0,0,1]])
    R_T=np.concatenate((R_T,tmp),axis=0) # appending the last row

    # inversing the camera's extrinsic matrix
    R_inverse=np.linalg.inv(R_T)
    R_matrix=R_inverse[:3,:3] # 3x3
    T_matrix=R_inverse[:3,3] # 3x1

    return R_matrix,T_matrix

def differences(rotq, tvec, rotq_gt, tvec_gt):
    nor_rotq = rotq / np.linalg.norm(rotq)
    nor_rotq_gt = rotq_gt / np.linalg.norm(rotq)
    dif_r = np.clip(np.sum(nor_rotq * nor_rotq_gt), 0, 1)

    d_r = np.degrees(np.arccos(2 * dif_r * dif_r - 1))
    d_t = np.linalg.norm(tvec-tvec_gt, 2)

    return d_r, d_t

# Process model descriptors
desc_df = average_desc(train_df, points3D_df)
kp_model = np.array(desc_df["XYZ"].to_list())
desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

differences_r = []
differences_t = []

df = pd.DataFrame(columns=['rotation', 'position'])

for i in trange(1,294):
    # Load quaery image 
    try:
        idx = i
        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        # retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
        rvec, tvec = pnpsolver((kp_query, desc_query),(kp_model, desc_model))

        r_pos, t_pos = find_camera_position(rvec,tvec)
        df = df.append({'rotation':r_pos, 'position':t_pos}, ignore_index=True)

        rotq = Rot.from_rotvec(rvec.reshape(1,3)).as_quat()
        tvec = tvec.reshape(1,3)

        # Get camera pose groudtruth 
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Computing absolute pose differences
        d_r, d_t = differences(rotq, tvec, rotq_gt, tvec_gt)
        differences_r.append(d_r)
        differences_t.append(d_t)
    except:
        continue

differences_r = np.array(differences_r)
differences_t = np.array(differences_t)

# median pose error
err_r = np.median(differences_r)
err_t = np.median(differences_t)

print("pose error: ", err_t, "rotation error: ", err_r)

# display(df)
filename = "./camera_position.pkl"
df.to_pickle(filename)



# normal p3p
# pose error:  4.50948783830821 rotation error:  109.21544155577249 
# p4p (like opencvc)
# pose error:  0.02988279282462082 rotation error:  0.7785275549845175