import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def abs_sobel_thresh(img, orient='x', kernel=3, thresh=(0, 255)):
    """
    Input: img, thresh
        img = input gray Image
        kernel = kernel size
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    if orient == 'x':
        x = 1
        y = 0
    else:
        x = 0
        y = 1
    # 1) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=kernel)
    # 2) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 3) Scale to 8-bits (0-255) then convert to type = np.unit8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 4) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return binary_output


# Magnitude of the Gradient
def mag_thresh(img, kernel=3, thresh=(0, 255)):
    """
    Input: img, thresh
        img = input gray Image
        kernel = kernel size
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    # 1) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # 2) Calculate the magnitude
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 3) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 4) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


# Direction of the Gradient
def dir_thresh(img, kernel=3, thresh=(0, np.pi/2)):
    """
    Input: img, thresh
        img = input gray Image
        kernel = kernel size
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    graddir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(graddir)
    binary_output[(graddir >= thresh[0]) & (graddir <= thresh[1])] = 1
    return binary_output


def hls_detect(img, thresh=(0, 255)):
    """
    Input: img, thresh
        img = input RGB Image
        thresh = gray image transform to binary image use
    Output: binary_output
        binary_output = output binary image
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel < thresh[1])] = 1
    return binary_output


def combing_sobel_schannel_thresh(img, kernel=3):
    """
    Is function combing Sobelx, Sobley, Magnitude and Direction Operator,
    Input: img, kernel
        img = input RGB Image
        kernel = kernel size
    Output: binary_output
        binary_output = output binary image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold x gradient
    sobalx_binary = abs_sobel_thresh(img_gray, orient='x', kernel=kernel, thresh=(30, 100))

    # Threshold color channel
    s_channel_binary = hls_detect(img, thresh=(90, 255))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sobalx_binary)
    combined_binary[(sobalx_binary == 1) | (s_channel_binary == 1)] = 1

    # plt.figure(figsize=(12, 12))
    # plt.subplot(3, 1, 1)
    # plt.title('Sobal X')
    # plt.imshow(sobalx_binary, cmap='gray')
    # plt.subplot(3, 1, 2)
    # plt.title('HLS S Channel')
    # plt.imshow(s_channel_binary, cmap='gray')
    # plt.subplot(3, 1, 3)
    # plt.title('Combine')
    # plt.imshow(combined_binary, cmap='gray')

    return combined_binary


def combing_smd_thresh(img, kernel=3):
    """
    Is function combing Sobelx, Sobley, Magnitude and Direction Operator,
    Input: img, kernel
        img = input RGB Image
        kernel = kernel size
    Output: binary_output
        binary_output = output binary image
    """
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', kernel=kernel, thresh=(30, 100))
    grady = abs_sobel_thresh(img, orient='y', kernel=kernel, thresh=(30, 100))
    mag_binary = mag_thresh(img, kernel=kernel, thresh=(20, 100))
    dir_binary = dir_thresh(img, kernel=kernel, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


if __name__ == '__main__':

    test_image = 'straight_lines1.jpg'
    test_image = '../test_images/test4.jpg'
    # Read image
    img = cv2.imread(test_image)
    # Image transform
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel Operator
    img_sobelx_out = abs_sobel_thresh(img_gray, orient='x', kernel=7, thresh=(30, 100))
    img_sobely_out = abs_sobel_thresh(img_gray, orient='y', kernel=7, thresh=(30, 100))
    # Magnitude Operator
    img_mag_out = mag_thresh(img_gray, kernel=7, thresh=(20, 100))
    # Direction Operator
    img_dir_out = dir_thresh(img_gray, kernel=7, thresh=(0.7, 1.3))
    # Combined Sobelx, Sobely, Magnitude, Direction the operator
    img_comb_out_1 = combing_smd_thresh(img_gray, kernel=7)

    # Customizing Figure Layouts
    fig = plt.figure(figsize=(16, 8))
    gs1 = gridspec.GridSpec(3, 2, right=0.48, wspace=0.1)

    ax1 = fig.add_subplot(gs1[0, 0])
    plt.title('Sobel X')
    plt.imshow(img_sobelx_out, cmap='gray')

    ax2 = fig.add_subplot(gs1[0, 1])
    plt.title('Sobel Y')
    plt.imshow(img_sobely_out, cmap='gray')

    ax3 = fig.add_subplot(gs1[1, 0])
    plt.title('Magnitude')
    plt.imshow(img_mag_out, cmap='gray')

    ax4 = fig.add_subplot(gs1[1, 1])
    plt.title('Direction')
    plt.imshow(img_dir_out, cmap='gray')

    ax5 = fig.add_subplot(gs1[2, :])
    plt.title('Combined')
    plt.imshow(img_comb_out_1, cmap='gray')

    # HLS S Channel detection
    img_s_output = hls_detect(img, thresh=(60, 255))
    # Combined Sobelx, HLS S Channel
    img_comb_out_2 = combing_sobel_schannel_thresh(img, kernel=7)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img_s_output, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.imshow(img_comb_out_2, cmap='gray')

    plt.show()
