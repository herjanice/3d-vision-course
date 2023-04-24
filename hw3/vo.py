import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import matplotlib.pyplot as plt


class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def load_camera(self, t, R):

        points = np.array([[0, 0, 1], [360, 0, 1], [360, 360, 1], [0, 360, 1]])
        points = np.linalg.pinv(self.K) @ points.T
        world = R @ points + t
        world = world.T
        world = np.concatenate((world, t.T), axis=0)

        model = o3d.geometry.LineSet()
        model.points = o3d.utility.Vector3dVector(world)
        model.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])

        color = np.array([1, 0, 0])
        model.colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))

        return model

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width = 1000, height=1000)
        queue = mp.Queue()

        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    
                    model=self.load_camera(t,R)
                    vis.add_geometry(model)

                    pass
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def extract_features(self, img1, img2):
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)

        points1 = np.array([kp1[m.queryIdx].pt for m in matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in matches])

        # img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_DEFAULT)
        # cv.imshow('match', img_draw_match)
        # cv.waitKey(0)

        return points1, points2

    def calculate_scale(self, prev_3d, curr_3d):
        prev_new = np.roll(prev_3d, shift=-5)
        curr_new = np.roll(curr_3d, shift=-5)
        ratios = (np.linalg.norm(curr_3d - curr_new,axis = -1)) / (np.linalg.norm(prev_3d - prev_new, axis = -1))

        ratio = np.median(ratios)
        return ratio

    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        prev_triangulated = None
        prev_ratio = None
        prev_points = None

        current_pos = np.zeros((3, 1), dtype=np.float64)
        current_rot = np.eye(3, dtype=np.float64)

        for i, frame_path in enumerate(self.frame_paths[1:]):
            img1 = cv.imread(self.frame_paths[i])
            img2 = cv.imread(frame_path)

            #TODO: compute camera pose here

            # Extract features between I_k+1 and I_k
            points1, points2 = self.extract_features(img1,img2)
            # points1 = cv.undistortPoints(points1, self.K, self.dist, None, self.K)
            # points2 = cv.undistortPoints(points2, self.K, self.dist, None, self.K)

            # Estimate the essential matrix E
            E, mask = cv.findEssentialMat(points1, points2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)

            # Decompose E to get relative pose
            _, R, t, mask, triangulated = cv.recoverPose(E, points1, points2, self.K, mask=mask, distanceThresh=50)

            # Get the 3d points through triangulation by cv.recoverPose
            triangulated = triangulated / triangulated[3]
            triangulated = triangulated[0:3]
            triangulated = triangulated.reshape((-1,3))

            # Getting only the inliers
            mask = mask.reshape((-1,1))
            p1 = []
            p2 = []
            tri = []
            for x in range(len(mask)):
                if mask[x][0] == 1:
                    p1.append(points1[x])
                    p2.append(points2[x])
                    tri.append(triangulated[x])

            points1 = np.array(p1)
            points2 = np.array(p2)
            triangulated = np.array(tri)  

            # Calculating the Ratio
            if i == 0: # the first one
                ratio = 1
                
            else:
                prev = []
                curr = []
                for i in range(len(points1)):
                    for j in range(len(prev_points)):
                        if (points1[i] == prev_points[j]).all:
                            prev.append(prev_triangulated[j])
                            curr.append(triangulated[i])

                if(len(prev)<=1):
                    ratio = 1
                    
                else:
                    ratio = self.calculate_scale(np.array(prev), np.array(curr))
                    ratio *= prev_ratio
                    if ratio > 2:
                        ratio = 2
                    elif ratio < 0.7:
                        ratio = 0.7

            current_pos += current_rot.dot(t) * ratio
            current_rot = R.dot(current_rot)
            queue.put((current_rot, current_pos))
            
            prev_ratio = ratio
            prev_points=points2
            prev_triangulated=triangulated
            
            cv.imshow('frame', img1)
            if cv.waitKey(30) == 27: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()