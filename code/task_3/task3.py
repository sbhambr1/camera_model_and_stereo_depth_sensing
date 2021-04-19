import numpy as np
import cv2
import glob
import argparse
import math 
from random import shuffle
import re

img_size = None

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

def ssc(keypoints, num_ret_points, tolerance, cols, rows):

    #Function taken from https://github.com/BAILOOL/ANMS-Codes

    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4) 
    sol2 = -round(float(exp1 - exp3) / exp4)  

    high = (
        sol1 if (sol1 > sol2) else sol2
    ) 
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if (
            width == prev_width or low > high
        ):  
            result_list = result  
            break

        c = width / 2  
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [
            [False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)
        ]
        result = []

        for i in range(len(keypoints)):
            row = int(
                math.floor(keypoints[i].pt[1] / c)
            ) 
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]: 
                result.append(i)
                row_min = int(
                    (row - math.floor(width / c))
                    if ((row - math.floor(width / c)) >= 0)
                    else 0
                )
                row_max = int(
                    (row + math.floor(width / c))
                    if ((row + math.floor(width / c)) <= num_cell_rows)
                    else num_cell_rows
                )
                col_min = int(
                    (col - math.floor(width / c))
                    if ((col - math.floor(width / c)) >= 0)
                    else 0
                )
                col_max = int(
                    (col + math.floor(width / c))
                    if ((col + math.floor(width / c)) <= num_cell_cols)
                    else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1 
        else:
            low = width + 1
        prev_width = width

    for i in range(len(result_list)):
        selected_keypoints.append(keypoints[result_list[i]])

    return selected_keypoints

def detect_features_and_match(save_img_dir_path, l_image, r_image, matrix_l, matrix_r, dist_l, dist_r):

    l_img = cv2.imread(l_image)
    r_img = cv2.imread(r_image)
    gray_l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    gray_r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

    l_img_size = (l_img.shape[1], l_img.shape[0])

    r_img_size = (r_img.shape[1], r_img.shape[0])

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_l, dist_l, None, None, l_img_size, cv2.CV_32FC1)
    l_undistorted_img = cv2.remap(gray_l_img, map_w, map_h, cv2.INTER_LINEAR)

    map_w, map_h = cv2.initUndistortRectifyMap(matrix_r, dist_r, None, None, r_img_size, cv2.CV_32FC1)
    r_undistorted_img = cv2.remap(gray_r_img, map_w, map_h, cv2.INTER_LINEAR)

    orb = cv2.ORB_create(nfeatures=2000)

    kp_l = orb.detect(l_undistorted_img, None)
    kp_r = orb.detect(r_undistorted_img, None)

    shuffle(kp_l)
    shuffle(kp_r)

    selected_kp_l = ssc(kp_l, 100, 0.1, l_img.shape[1], l_img.shape[0])
    selected_kp_r = ssc(kp_r, 100, 0.1, r_img.shape[1], r_img.shape[0])

    l_kp_img = cv2.drawKeypoints(l_undistorted_img, selected_kp_l, None, color=(255, 0, 0), flags = 0)
    r_kp_img = cv2.drawKeypoints(r_undistorted_img, selected_kp_r, None, color=(255, 0, 0), flags = 0)

    cv2.imwrite(save_img_dir_path + '/' + 'ORB_Left.png', l_kp_img)
    cv2.imwrite(save_img_dir_path + '/' + 'ORB_Right.png', r_kp_img)

    selected_kp_l, selected_des_l = orb.compute(l_kp_img, selected_kp_l)
    selected_kp_r, selected_des_r = orb.compute(r_kp_img, selected_kp_r)
         
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(selected_des_l, selected_des_r)
    matches = sorted(matches, key= lambda x:x.distance)

    matched_img = cv2.drawMatches(l_kp_img, selected_kp_l, r_kp_img, selected_kp_r, matches[:30], None)

    cv2.imwrite(save_img_dir_path +'/' + 'matched_feature_points.png', matched_img)

    return [selected_kp_l, selected_kp_r]
    
def triangulate_points(pose_1, pose_2, kp_l, kp_r):

    if len(kp_l)<len(kp_r):
        kp_r = cv2.KeyPoint_convert(kp_r[:len(kp_l)])
        kp_l = cv2.KeyPoint_convert(kp_l)
    else:
        kp_l = cv2.KeyPoint_convert(kp_l[:len(kp_r)])
        kp_r = cv2.KeyPoint_convert(kp_r) 

    triangulated = cv2.triangulatePoints(pose_1[:3,:], pose_2[:3,:], kp_l[:2], kp_r[:2])

    return triangulated/triangulated[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Sparse Depth Triangulation!')
    parser.add_argument('--images_dir', type=str, required=True, help='Image Directory Path')
    parser.add_argument('--l_prefix', type=str, required=True, help='Image prefix - left')
    parser.add_argument('--r_prefix', type=str, required=True, help='Image prefix - right')
    parser.add_argument('--detect_features_img_l', type=str, required=True, help='l_image to be generated after drawing key points')
    parser.add_argument('--detect_features_img_r', type=str, required=True, help='r_image to be generated after drawing key points')
    parser.add_argument('--save_images_dir', type=str, required=True, help='Save Image Directory Path')

    args = parser.parse_args()
    print('Arguments parsed')
    obj_points, img_points_l, img_points_r, ret, matrix_l, dist_l, matrix_r, dist_r = calibrate_camera(args.images_dir, args.l_prefix, args.r_prefix)
    print('Camera calibrated')
    retval, matrix_1, dist_1, matrix_2, dist_2, R, T, E, F = stereo_calibrate(obj_points, img_points_l, img_points_r, img_size, matrix_l, dist_l, matrix_r, dist_r)
    print('Stereo calibration done')
    rot_1, rot_2, pose_1, pose_2, Q, roi_left, roi_right = rectify_stereo_camera(matrix_1, dist_1, matrix_2, dist_2, R, T)
    print('Stereo camera rectified')
    kp_l, kp_r = detect_features_and_match(args.save_images_dir, args.detect_features_img_l, args.detect_features_img_r, matrix_l, matrix_r, dist_l, dist_r)
    print('Key points drawn and matched')
    triangulated = triangulate_points(pose_1, pose_2, kp_l, kp_r)
    print('Points triangulated')