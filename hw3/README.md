# Homework 3: Visual Odometry

The process, results and finding of this homework could be found in [here](https://github.com/herjanice/3d-vision-course/blob/main/hw3/report.pdf)

Task: Implement a VO based on two-view epipolar geometry

Input: Image sequence and the camera intrinsic
Output: Sequential global camera pose (w.r.t the coordinate system of the first frame)

**Step 1: Camera Calibration**
The program for camera calibration is provided on [camera_calibration.py](https://github.com/herjanice/3d-vision-course/blob/main/hw3/camera_calibration.py)
Use the program to obtain camera intrinsic matrix and the distortion coefficients.

**Step 2: Feature Matching**
Recommendation: Use ORB as feature extractor
- 10x faster than SIFT
- binary descriptor
- orientation and scale invariance
- compute Hamming distance for binary feature matching

**Step 3: Pose from Epipolar Geometry**
Check class slide 21 (Hint: use cv2.findEssentialMat, cv2.recoverPose, cv2.triangulatePoints)

**Step 4: Results Visualization**
Draw the matched (tracked) point on current image and update the new camera pose in Open3D window. Template for showing current image and visualizing camera trajectory in open3D is provided
