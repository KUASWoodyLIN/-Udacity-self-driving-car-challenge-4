import os
import random
import pickle
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt


IMAGE_PATH = '../camera_cal/'
images_path = glob(IMAGE_PATH + '*.jpg')
file = 'wide_dist_pickle.p'


def found_chessboard():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) (x,y,z)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)   # x,y,coordinates

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for img_path in images_path:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('image', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints, imgpoints


def camera_cal(objpoints, imgpoints):
    # random chose image to test
    random_chose = random.randint(0, len(images_path)-1)
    img_test = images_path[random_chose]

    # Test undistortion on an image
    img = cv2.imread(img_test)
    img_size = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('Undistorted Image')
    plt.imshow(dst)
    plt.show()
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    if not os.path.exists(file):
        print('Pickle File {} is not exists, create one now.'.format(file))
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open("wide_dist_pickle.p", "wb"))

    return mtx, dist


def read_camera_cal_file():
    with open(file, 'rb') as f:
        dump = pickle.load(f)
    return dump['mtx'], dump['dist']


if __name__ == '__main__':
    if not os.path.exists(file):
        objpoints, imgpoints = found_chessboard()
        mtx, dist = camera_cal(objpoints, imgpoints)
    else:
        print('Get parameter from pickle file')
        mtx, dist = read_camera_cal_file()
