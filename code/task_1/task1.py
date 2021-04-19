import numpy as np
import cv2
import glob
import argparse

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

def calibrate_camera(dir_path, prefix_name, size = 0.0254, width=9, height=6):
    o_points = np.zeros((height*width, 3), np.float32)
    o_points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    o_points = o_points * size

    obj_points = []
    img_points = []

    images = glob.glob(dir_path + '/' + prefix_name +'*.png')

    for img_name in images:
        img = cv2.imread(img_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray_img, (width, height), None)

        if ret is True:
            obj_points.append(o_points)

            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    
    return [ret, matrix, dist, rvecs, tvecs, img]

def remap_image(save_dir_path, img_path, matrix, dist, img_corners):

    cv2.imwrite(save_dir_path + '/' + 'chessboard_corners.png', img_corners)

    img = cv2.imread(img_path)
    img_size = (img.shape[1], img.shape[0])
    map_w, map_h = cv2.initUndistortRectifyMap(matrix, dist, None, None, img_size, cv2.CV_32FC1)
    distort = cv2.remap(img, map_w, map_h, cv2.INTER_LINEAR)

    cv2.imwrite(save_dir_path + '/' + 'calibrated_image.png', distort)

def save_parameters(dir_path, matrix, dist):
    saved_params = cv2.FileStorage(dir_path, cv2.FILE_STORAGE_WRITE)
    saved_params.write('matrix_K', matrix)
    saved_params.write('distort_coeff', dist)
    saved_params.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Calibrating camera and remapping image!')
    parser.add_argument('--images_dir', type=str, required=True, help='Image Directory Path')
    parser.add_argument('--prefix', type=str, required=True, help='Image prefix - left/right')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save required parameters')
    parser.add_argument('--remap_img', type=str, required=True, help='image to be remapped after undistorting')
    parser.add_argument('--save_images_dir', type=str, required=True, help='Save Image Directory Path')

    args = parser.parse_args()
    print('Arguments parsed')
    ret, matrix, dist, rvecs, tvecs, img = calibrate_camera(args.images_dir, args.prefix)
    print('Camera calibrated')
    remap_image(args.save_images_dir, args.remap_img, matrix, dist, img)
    print('Image remapped')
    save_parameters(args.save_file, matrix, dist)
    print('Parameters saved')
    print('Camera calibration is complete!')
