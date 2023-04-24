# Homework 1:
The process, results and finding of this homework could be found in [here](https://github.com/herjanice/3d-vision-course/blob/main/hw1/report.pdf)

## Problem 1: Homography Estimation

Task:  Estimate homography between two given images by finding corresponding points, computing the homography matrix, and evaluating the estimation error.

**Part 1: Feature Matching**
1. Perform local feature detection on both images
2. Find correspondences between anchor and target images using descriptors
3. Reject outliers using ratio test or manual comparison
4. Select top k pairs from matching results, with k as 4, 8, or 20
(BONUS) using alternative local feature, such as SuperPoint

**Part 2: Direct Linear Transform**
1. Estimate the homography between anchor and target images for each k value using direct linear transform.
2. Compute the reprojection error using ground truth matching pairs.

**Part 3: Normalized Direct Linear Transform**
1. Similar to Part 2, use normalized direct linear transform to estimate the homography between the anchor and target images.
2. Compute the error and results, and compare them with those from Part 2 to identify any differences.
(BONUS) Experiment with alternative methods or techniques that could potentially improve the DLT or normalized DLT approach.

## Problem 2: Document Rectification

Task:  Recover the original geometry of a document image distorted by perspective transformation caused by camera capture.

**Part 1: Capture Document, Mark Corner Points**
1. Capture an image of a document you want to rectify
2. Mark the corner points of the image


**Part 2: Homography Estimation, Warp Image**
1. Compute the homography
2. Implement bilinear interpolation for image warping
