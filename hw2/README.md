# Homework 2:

The process, results and finding of this homework could be found in [here](https://github.com/herjanice/3d-vision-course/blob/main/hw2/report.pdf)

## Problem 1: 2D-3D Matching

**Part 1:**
Task: For each image, compute the camera pose with respect to world coordinate.  Implement an algorithm for the PnP (Perspective-n-Point) problem, with the expected solution being P3P + RANSAC. Other potential solutions, such as DLT, EPnP, AP3P, etc., may also be utilized.

1. Find 2D-3D correspondence by descriptor matching
2. Solve camera pose

**Part 2**
Task: Compute the median pose error (translation, rotation) for each camera pose calculated on Part1 with respect to ground truth camera pose.

**Part 3**
Task: Plot the trajectory and camera poses along with 3D point cloud model using Open3D

## Problem 2: Augmented Reality

**Part 1**
Task: Create an Augmented Reality video by placing a virtual cube in the validation image sequences using camera intrinsic and extrinsic parameters. The cube should be represented as a point set with colored surfaces, and a painter's algorithm should be implemented to determine the order of drawing.

Notes:
- Occlusion of the virtual cube does not need to be considered.

