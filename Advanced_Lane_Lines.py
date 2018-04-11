import os
import random
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from Udacity_self_driving_car_challenge_4.image_processing.calibration import camera_cal, found_chessboard, read_camera_cal_file
from Udacity_self_driving_car_challenge_4.image_processing.edge_detection import combing_color_thresh
from Udacity_self_driving_car_challenge_4.image_processing.find_lines import conv_sliding_search, histogram_search
# The goals / steps of this project are the following:
#
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.   ok
# Apply a distortion correction to raw images.                                      ok
# Use color transforms, gradients, etc., to create a thresholded binary image.      ok
# Apply a perspective transform to rectify binary image ("birds-eye view").         ok
# Detect lane pixels and fit to find the lane boundary.                             ok
# Determine the curvature of the lane and vehicle position with respect to center.  ok
# Warp the detected lane boundaries back onto the original image.                   ok
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

ROOT_PATH = os.getcwd()
IMAGE_TEST_DIR = os.path.join(ROOT_PATH, 'test_images')
IMAGE_OUTPUT_DIR = os.path.join(ROOT_PATH, 'output_images')
VIDEO_OUTPUT_DIR = os.path.join(ROOT_PATH, 'output_video')
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
src = np.float32([(596, 447), (683, 447), (1120, 720), (193, 720)])   # Longer one
#src = np.float32([(578, 460), (704, 460), (1120, 720), (193, 720)])
dst = np.float32([(offset-300, 0), (offset+300, 0), (offset+300, 720), (offset-300, 720)])
perspective_M = cv2.getPerspectiveTransform(src, dst)
inver_perspective_M= cv2.getPerspectiveTransform(dst, src)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def process_image(image):
    # Apply a distortion correction to raw images.
    image = cv2.undistort(image, mtx, dist, None, None)

    # Use color transforms, gradients to find the object edge and change into binary image
    image_binary = combing_color_thresh(image)
    plt.figure()
    plt.imshow(image_binary, cmap='gray')

    # Transform image to bird view
    image_bird_view = cv2.warpPerspective(image_binary, perspective_M, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # find the road lines, curvature and distance between car_center and road_center
    color_warp, curv, center, left_or_right = histogram_search(image_bird_view)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inver_perspective_M, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    img_out = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # Add description on images
    text1 = "Radius of Curature = {:.2f}(m)".format(curv)
    text2 = "Vehicle is {:.3f}m {} of center".format(abs(center), left_or_right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_out, text1, (50, 50), font, 1.5, color=(255, 255, 255), thickness=3)
    cv2.putText(img_out, text2, (50, 100), font, 1.5, color=(255, 255, 255), thickness=3)
    return img_out


def test_image():
    # random chose image to test
    random_chose = random.randint(0, len(IMAGES_PATH)-1)
    img_test = IMAGES_PATH[random_chose]
    print(img_test)
    img = cv2.imread('./test_images/test5.jpg')

    img_out = process_image(img)

    # Converter BGR -> RGB for plt show
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.title('Output Image')
    plt.imshow(img_out, cmap='gray')
    plt.show()


def test_images():
    for path in IMAGES_PATH:
        img = cv2.imread(path)
        img_out = process_image(img)
        img_out_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.split(path)[-1].split('.')[0] + '.png')
        cv2.imwrite(img_out_path, img_out)


def test_video():
    video_file = 'project_video.mp4'
    video_output_file = os.path.join(VIDEO_OUTPUT_DIR, video_file.split('.')[0] + '.avi')

    # Video save
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_file, fourcc, 20, (1280, 720))

    # Video read
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_out = process_image(frame)

            out.write(img_out)
            cv2.imshow('frame', img_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_video()
