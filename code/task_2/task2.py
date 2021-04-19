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
    
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    flags |= cv2.CALIB_ZERO_TANGENT_DIST

    retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r, matrix_l, dist_l, matrix_r, dist_r, (640, 480), criteria= criteria_stereo_cal, flags=flags)

    return [retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F]

def save_stereo_calibration_parameters(dir_path, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F):

    saved_params = cv2.FileStorage(dir_path, cv2.FILE_STORAGE_WRITE)
    saved_params.write('matrix_1', matrix_1)
    saved_params.write('distort_coeff_1', dist_1)
    saved_params.write('matrix_2', matrix_2)
    saved_params.write('distort_coeff_2', dist_2)
    saved_params.write('rotation_matrix', R)
    saved_params.write('translation_matrix', T)
    saved_params.write('essential_matrix', E)
    saved_params.write('fundamental_matrix', F)
    saved_params.release()

def check_calibration(dir_path, l_prefix_name, r_prefix_name, matrix_1, dist_1, matrix_2, dist_2, size = 0.0254, width=9, height=6):

    l_images = glob.glob(dir_path + '/' + l_prefix_name +'*.png')
    r_images = glob.glob(dir_path + '/' + r_prefix_name +'*.png')

    for i,_ in enumerate(l_images):
        l_img = cv2.imread(l_images[i])
        r_img = cv2.imread(r_images[i])
        gray_l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        gray_r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

        l_ret, l_corners = cv2.findChessboardCorners(gray_l_img, (width, height), None)
        r_ret, r_corners = cv2.findChessboardCorners(gray_r_img, (width, height), None)

    l_corners = np.reshape(l_corners, (54, 2))
    r_corners = np.reshape(r_corners, (54, 2))

    homogenize_matrix = np.zeros((3,1))
    homog_matrix_1 = np.concatenate((matrix_1, homogenize_matrix), axis=1)
    homog_matrix_2 = np.concatenate((matrix_2, homogenize_matrix), axis=1)

    points_l_norm = cv2.undistortPoints(l_corners, matrix_1, dist_1)
    points_r_norm = cv2.undistortPoints(r_corners, matrix_2, dist_2)

    plottable = cv2.triangulatePoints(homog_matrix_1, homog_matrix_2, np.transpose(points_l_norm), np.transpose(points_r_norm))

def rectify_stereo_camera(matrix_1, dist_1, matrix_2, dist_2, R, T):

    rotation_1, rotation_2, pose_1, pose_2, Q, roi_left, roi_right = cv2.stereoRectify(matrix_1, dist_1, matrix_2, dist_2, (640, 480), R, T, flags=cv2.CALIB_ZERO_DISPARITY)

    return [rotation_1, rotation_2, pose_1, pose_2, Q]

def save_stereo_rectification_parameters(dir_path, rotation_1, rotation_2, pose_1, pose_2, Q):

    saved_params = cv2.FileStorage(dir_path, cv2.FILE_STORAGE_WRITE)
    saved_params.write('rotation_1', rotation_1)
    saved_params.write('rotation_2', rotation_2)
    saved_params.write('pose_1', pose_1)
    saved_params.write('pose_2', pose_2)
    saved_params.write('disparity_depth_matrix', Q)
    saved_params.release()

def generate_undistored_rectified_image_l(dir_path, img_path, matrix_l, dist_l, matrix_1, dist_1, rot_1, pose_1):

    img = cv2.imread(img_path)
    img_size =(img.shape[1], img.shape[0])

    ret, corners = cv2.findChessboardCorners(img, (9,6), None)
    img_with_corners = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    cv2.imwrite(dir_path + '/' + 'original_l_with_corners.png', img_with_corners)

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_l, dist_l, None, None, img_size, cv2.CV_32FC1)
    undistorted = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    ret, corners = cv2.findChessboardCorners(undistorted, (9,6), None)
    undistorted_with_corners = cv2.drawChessboardCorners(undistorted, (9,6), corners, ret)

    cv2.imwrite(dir_path + '/' + 'undistorted_image_l.png', undistorted)
    cv2.imwrite(dir_path + '/' + 'undistorted_image_l_with_corners.png', undistorted_with_corners)
    
    map_w, map_h = cv2.initUndistortRectifyMap(matrix_1, dist_1, rot_1, pose_1, (640, 480), cv2.CV_32FC1)
    rectified = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    ret, corners = cv2.findChessboardCorners(rectified, (9,6), None)
    rectified_with_corners = cv2.drawChessboardCorners(rectified, (9,6), corners, ret)

    cv2.imwrite(dir_path + '/' + 'undistorted_rectified_image_l.png', rectified)
    cv2.imwrite(dir_path + '/' + 'undistorted_image_l_with_corners.png', rectified_with_corners)

def generate_undistored_rectified_image_r(dir_path, img_path, matrix_r, dist_r, matrix_2, dist_2, rot_2, pose_2):

    img = cv2.imread(img_path)
    img_size =(img.shape[1], img.shape[0])

    ret, corners = cv2.findChessboardCorners(img, (9,6), None)
    img_with_corners = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    cv2.imwrite(dir_path + '/' + 'original_r_with_corners.png', img_with_corners)

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_r, dist_r, None, None, img_size, cv2.CV_32FC1)
    undistorted = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    ret, corners = cv2.findChessboardCorners(undistorted, (9,6), None)
    undistorted_with_corners = cv2.drawChessboardCorners(undistorted, (9,6), corners, ret)

    cv2.imwrite(dir_path + '/' + 'undistorted_image_r.png', undistorted)
    cv2.imwrite(dir_path + '/' + 'undistorted_image_r_with_corners.png', undistorted_with_corners)

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_2, dist_2, rot_2, pose_2, (640, 480), cv2.CV_32FC1)
    rectified = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    ret, corners = cv2.findChessboardCorners(rectified, (9,6), None)
    rectified_with_corners = cv2.drawChessboardCorners(rectified, (9,6), corners, ret)

    cv2.imwrite(dir_path + '/' + 'undistorted_rectified_image_r.png', rectified)
    cv2.imwrite(dir_path + '/' + 'undistorted_image_r_with_corners.png', rectified_with_corners)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stereo calibration and rectification!')
    parser.add_argument('--images_dir', type=str, required=True, help='Image Directory Path')
    parser.add_argument('--l_prefix', type=str, required=True, help='Image prefix - left')
    parser.add_argument('--r_prefix', type=str, required=True, help='Image prefix - right')
    parser.add_argument('--save_file_calibration', type=str, required=True, help='YML file to save required parameters')
    parser.add_argument('--save_file_rectification', type=str, required=True, help='YML file to save required parameters')
    parser.add_argument('--undistort_rectified_img_l', type=str, required=True, help='l_image to be generated after undistorting and rectifying')
    parser.add_argument('--undistort_rectified_img_r', type=str, required=True, help='r_image to be generated after undistorting and rectifying')
    parser.add_argument('--save_images_dir', type=str, required=True, help='Save Image Directory Path')

    args = parser.parse_args()
    print('Arguments parsed')
    obj_points, img_points_l, img_points_r, ret, matrix_l, dist_l, matrix_r, dist_r = calibrate_camera(args.images_dir, args.l_prefix, args.r_prefix)
    print('Camera calibrated')
    retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F = stereo_calibrate(obj_points, img_points_l, img_points_r, img_size, matrix_l, dist_l, matrix_r, dist_r)
    print('Stereo calibration done')
    save_stereo_calibration_parameters(args.save_file_calibration, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F)
    print('Calibration parameters saved')
    check_calibration(args.images_dir, args.l_prefix, args.r_prefix, matrix_1, dist_1, matrix_2, dist_2)
    print('Checked calibration')
    rot_1, rot_2, pose_1, pose_2, Q = rectify_stereo_camera(matrix_1, dist_1, matrix_2, dist_2, R, T)
    print('Stereo camera rectified')
    save_stereo_rectification_parameters(args.save_file_rectification, rot_1, rot_2, pose_1, pose_2, Q)
    print('Rectification parameters saved')
    generate_undistored_rectified_image_l(args.save_images_dir, args.undistort_rectified_img_l, matrix_l, dist_l, matrix_1, dist_1, rot_1, pose_1)
    generate_undistored_rectified_image_r(args.save_images_dir, args.undistort_rectified_img_r, matrix_r, dist_r, matrix_2, dist_2, rot_2, pose_2)
    print('Corrected image pair generated')
