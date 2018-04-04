import os
import random
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt

from Udacity_self_driving_car_challenge_4.image_processing.calibration import camera_cal, found_chessboard, read_camera_cal_file
from Udacity_self_driving_car_challenge_4.image_processing.edge_detection import combing_sobel_schannel_thresh
from Udacity_self_driving_car_challenge_4.image_processing.find_lines import sliding_search
# The goals / steps of this project are the following:
#
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.   ok
# Apply a distortion correction to raw images.                                  ok
# Use color transforms, gradients, etc., to create a thresholded binary image.  ok
# Apply a perspective transform to rectify binary image ("birds-eye view").     ok
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

ROOT_PATH = os.getcwd()
IMAGE_TEST_DIR = os.path.join(ROOT_PATH, 'test_images')
IMAGE_PROCESSING_PATH = os.path.join(ROOT_PATH, 'image_processing')
WIDE_DIST_FILE = os.path.join(IMAGE_PROCESSING_PATH, 'wide_dist_pickle.p')
IMAGES_PATH = glob(IMAGE_TEST_DIR + '/*.jpg')

# Load cameraMatrix and distCoeffs parameter
if not os.path.exists(WIDE_DIST_FILE):
    objpoints, imgpoints = found_chessboard()
    mtx, dist = camera_cal(objpoints, imgpoints)
else:
    print('Get parameter from pickle file')
    mtx, dist = read_camera_cal_file(WIDE_DIST_FILE)

# Get Perspective Transform Parameter
offset = 1280 / 2
src = np.float32([(596, 447), (683, 447), (1120, 720), (193, 720)])
dst = np.float32([(offset-300, 0), (offset+300, 0), (offset+300, 720), (offset-300, 720)])
perspective_M = cv2.getPerspectiveTransform(src, dst)


def process_image(image):
    image = cv2.undistort(image, mtx, dist, None, None)
    image_binary = combing_sobel_schannel_thresh(image, kernel=7)
    image_bird_view = cv2.warpPerspective(image_binary, perspective_M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    image_out = sliding_search(image_bird_view)

    histogram = np.sum(image_bird_view[image_bird_view.shape[0] // 2:, :], axis=0)
    plt.plot(histogram)
    result = image_out  # remove

    return result, image_bird_view


if __name__ == '__main__':
    # random chose image to test
    random_chose = random.randint(0, len(IMAGES_PATH)-1)
    img_test = IMAGES_PATH[random_chose]
    print(img_test)
    img = plt.imread('./test_images/test6.jpg')
    #img = plt.imread(img_test)

    img_out, image_bird_view = process_image(img)

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(3, 1, 2)
    plt.title('Undistorted Image')
    plt.imshow(img_out, cmap='gray')
    plt.subplot(3, 1, 3)
    plt.title('Undistorted Image')
    plt.imshow(image_bird_view, cmap='gray')
    plt.show()
