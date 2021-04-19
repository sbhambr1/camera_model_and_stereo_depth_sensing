import numpy as np
import cv2
import glob
import argparse
import re

img_size = (640, 480)

criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

criteria_stereo_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def calibrate_camera(dir_path, l_prefix_name, r_prefix_name, size = 0.0254, width=9, height=6):

    o_points = np.zeros((height*width, 3), np.float32)
    o_points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    o_points = o_points * size

    obj_points = []
    img_points_l = []
    img_points_r = []

    l_images = glob.glob(dir_path + '/' + l_prefix_name +'*.png')
    r_images = glob.glob(dir_path + '/' + r_prefix_name +'*.png')

    l_images = sorted(l_images, key=numerical_sort)
    r_images = sorted(r_images, key=numerical_sort)

    for i,_ in enumerate(l_images):
        l_img = cv2.imread(l_images[i])
        r_img = cv2.imread(r_images[i])
        gray_l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        gray_r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

        l_ret, l_corners = cv2.findChessboardCorners(gray_l_img, (width, height), None)
        r_ret, r_corners = cv2.findChessboardCorners(gray_r_img, (width, height), None)

        obj_points.append(o_points)

        if l_ret is True:

            corners_l = cv2.cornerSubPix(gray_l_img, l_corners, (11, 11), (-1, -1), criteria_cal)
            img_points_l.append(corners_l)

            l_img = cv2.drawChessboardCorners(l_img, (width, height), l_corners, l_ret)

        if r_ret is True:

            corners_r = cv2.cornerSubPix(gray_r_img, r_corners, (11, 11), (-1, -1), criteria_cal)
            img_points_r.append(corners_r)

            r_img = cv2.drawChessboardCorners(r_img, (width, height), r_corners, r_ret)

    ret, matrix_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_l, gray_l_img.shape[::-1], None, None)

    ret, matrix_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_r, gray_r_img.shape[::-1], None, None)

    return [obj_points, img_points_l, img_points_r, ret, matrix_l, dist_l, matrix_r, dist_r]

def stereo_calibrate(obj_points, img_points_l, img_points_r, img_size, matrix_l, dist_l, matrix_r, dist_r):
    
    flag = 0

    flag |= cv2.CALIB_FIX_INTRINSIC

    retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r, matrix_l, dist_l, matrix_r, dist_r, img_size, criteria= criteria_stereo_cal)

    return [retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F]

def rectify_stereo_camera(matrix_1, dist_1, matrix_2, dist_2, R, T):

    rotation_1, rotation_2, pose_1, pose_2, Q, roi_left, roi_right = cv2.stereoRectify(matrix_1, dist_1, matrix_2, dist_2, (9, 6), R, T)

    return [rotation_1, rotation_2, pose_1, pose_2, Q, roi_left, roi_right]

def generate_undistored_rectified_image_l(dir_path, img_path, matrix_l, dist_l, matrix_1, dist_1, rot_1, pose_1, roi_left):

    img = cv2.imread(img_path)
    img_size =(img.shape[1], img.shape[0])

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_l, dist_l, None, None, img_size, cv2.CV_32FC1)
    undistorted = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_1, dist_1, rot_1, pose_1, img_size, cv2.CV_32FC1)
    rectified = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    cv2.imwrite(dir_path + '/' + 'undistorted_rectified_image_l.png', rectified)

    return rectified

def generate_undistored_rectified_image_r(dir_path, img_path, matrix_r, dist_r, matrix_2, dist_2, rot_2, pose_2, roi_right):

    img = cv2.imread(img_path)
    img_size =(img.shape[1], img.shape[0])

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_r, dist_r, None, None, img_size, cv2.CV_32FC1)
    undistorted = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_2, dist_2, rot_2, pose_2, img_size, cv2.CV_32FC1)
    rectified = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    cv2.imwrite(dir_path + '/' + 'undistorted_rectified_image_r.png', rectified)

    return rectified

def compute_disparity(save_dir_path, rectified_l, rectified_r, Q):

    window_size = 7
    min_disp = 0
    max_disp = 64
    num_disp = max_disp - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize= window_size,
        preFilterCap=63,
        uniquenessRatio = 15,
        speckleWindowSize = 10,
        speckleRange = 1,
        disp12MaxDiff = 20,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    left_matcher = stereo
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    l = 70000
    s = 1.2

    disparity_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    disparity_filter.setLambda(l)
    disparity_filter.setSigmaColor(s)

    d_l = left_matcher.compute(rectified_l, rectified_r)
    d_r = right_matcher.compute(rectified_r, rectified_l)

    d_l = np.int16(d_l)
    d_r = np.int16(d_r)
    
    d_filter = disparity_filter.filter(d_l, rectified_l, None, d_r)

    disparity = cv2.normalize(d_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(save_dir_path + 'disparity_map.png', disparity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stereo calibration and rectification!')
    parser.add_argument('--images_dir', type=str, required=True, help='Image Directory Path')
    parser.add_argument('--l_prefix', type=str, required=True, help='Image prefix - left')
    parser.add_argument('--r_prefix', type=str, required=True, help='Image prefix - right')
    parser.add_argument('--undistort_rectified_img_l', type=str, required=True, help='l_image to be generated after undistorting and rectifying')
    parser.add_argument('--undistort_rectified_img_r', type=str, required=True, help='r_image to be generated after undistorting and rectifying')
    parser.add_argument('--save_images_dir', type=str, required=True, help='Save Image Directory Path')

    args = parser.parse_args()
    print('Arguments parsed')
    obj_points, img_points_l, img_points_r, ret, matrix_l, dist_l, matrix_r, dist_r = calibrate_camera(args.images_dir, args.l_prefix, args.r_prefix)
    print('Camera calibrated')
    retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F = stereo_calibrate(obj_points, img_points_l, img_points_r, img_size, matrix_l, dist_l, matrix_r, dist_r)
    print('Stereo calibration done')
    rot_1, rot_2, pose_1, pose_2, Q, roi_left, roi_right = rectify_stereo_camera(matrix_1, dist_1, matrix_2, dist_2, R, T)
    print('Stereo camera rectified')
    rectified_l = generate_undistored_rectified_image_l(args.save_images_dir, args.undistort_rectified_img_l, matrix_l, dist_l, matrix_1, dist_1, rot_1, pose_1, roi_left)
    rectified_r = generate_undistored_rectified_image_r(args.save_images_dir, args.undistort_rectified_img_r, matrix_r, dist_r, matrix_2, dist_2, rot_2, pose_2, roi_right)
    print('Corrected image pair generated')
    compute_disparity(args.save_images_dir, rectified_l, rectified_r, Q)
    print('Disparity computed')

