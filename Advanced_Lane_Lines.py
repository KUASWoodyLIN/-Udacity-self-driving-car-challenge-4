import os
import random
from glob import glob

import numpy
import cv2
import matplotlib.pyplot as plt


from image_processing.calibration import camera_cal, found_chessboard, read_camera_cal_file, WIDE_DIST_FILE


# The goals / steps of this project are the following:
#
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

ROOT_PATH = os.getcwd()
IMAGE_TEST_DIR = os.path.join(ROOT_PATH, 'test_images')
IMAGES_PATH = glob(IMAGE_TEST_DIR + '/*.jpg')

# Load cameraMatrix and distCoeffs parameter
if not os.path.exists(WIDE_DIST_FILE):
    objpoints, imgpoints = found_chessboard()
    mtx, dist = camera_cal(objpoints, imgpoints)
else:
    print('Get parameter from pickle file')
    mtx, dist = read_camera_cal_file()

def process_image(image):
    cv2.undistort(image, mtx, dist, None, None)

    result = image  # remove
    return result


if __name__ == '__main__':
    # random chose image to test
    random_chose = random.randint(0, len(IMAGES_PATH)-1)
    img_test = IMAGES_PATH[random_chose]
    img = plt.imread(img_test)

    img_out = process_image(img)
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('Undistorted Image')
    plt.imshow(img_out)
    plt.show()