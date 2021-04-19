Task 1:

Aim:​ The main aim of this task is to calibrate a pinhole camera model using a few images.

Results:
After calibrating the pinhole camera using a set of 10 images taken from both the left and right
camera, we obtain the left and right camera parameters which includes the camera intrinsic
matrix and the distort coefficient matrix. The following are the images which we obtain from
undistorting the images using the calibrated camera parameters.

● Data format of the camera parameter files: ​ ‘.yml’

Running the code:

Libraries imported: ​ numpy, cv2, glob, argparse

After setting the folder as the root directory, run the following command with suitable
arguments:

(An example has been provided for reference for left_2.png)

python3 ./code/task_1/task1.py --images_dir ./images/task_1/
./parameters/left_camera_intrinsics.yml
--remap_img
--save_images_dir ./output/task_1/
--prefix left --save_file
./images/task_1/left_2.png


Task 2:

Aim:​ The main aim of this task is to calibrate a stereo camera and then rectify it.

Results:
After the camera calibration has been done, we find the translation and rotation matrices for the
left and right cameras using the 3D-to-2D point correspondences. This is known as the stereo
calibration process which is then followed by the stereo rectification part. Through stereo
calibration, we are able to obtain the Rotation, Translation, Essential and Fundamental matrix. In
the second part of stereo rectification, we get the rotation matrices R1 and R2 along with the
camera poses P1 and P2, and also the disparity matrix Q. In the following figure, we see the
annotated chessboard corners in the original images, followed by the undistorted images and in
the end the corresponding undistorted and rectified images.

● Data format of the camera parameter files: ​ ‘.yml’
● Please note: The camera parameters have been recalculated for reusability of the code on
a new set of images as well.

Running the code:

Libraries imported: ​ numpy, cv2, glob, argparse, re

After setting the folder as the root directory, run the following command with suitable
arguments:
(An example has been provided for reference for left_1.png and right_1.png)

python3 ./code/task_2/task2.py --images_dir ./images/task_1/ --l_prefix left --r_prefix right
--save_file_calibration
./parameters/stereo_caliberation.yml
--save_file_rectification
./parameters/stereo_rectifcation.yml
--undistort_rectified_img_l ./images/task_2/left_1.png
--undistort_rectified_img_r ./images/task_2/right_1.png --save_images_dir ./output/task_2/

Task 3:

Aim: The main aim of this task is to calculate the rotation and translation relationship of two
views obtained by the same camera at different poses.

Results:
After the stereo calibration and rectification step, we perform the sparse depth triangulation. In
this step, we calculate the ORB feature points for the left and right images and then match the
corresponding key points obtained from the two images. Due to the different number and depth
of objects present in the given images, we show the results for the following two different views.

● Please note: The camera parameters have been recalculated for reusability of the code on
a new set of images as well.

Running the code:

Libraries imported: ​ numpy, cv2, glob, argparse, math, random, re

After setting the folder as the root directory, run the following command with suitable
arguments:

(An example has been provided for reference for left_7.png and right_7.png)

python3 ./code/task_3/task3.py --images_dir ./images/task_1/ --l_prefix left --r_prefix right
--detect_features_img_l
./images/task_3_and_4/left_7.png
--detect_features_img_r
./images/task_3_and_4/right_7.png --save_images_dir ./output/task_3/

Task 4:

Aim:​ The main aim of this task is to find the disparity map for a set of images.

Results:
After the sparse depth triangulation step, we find the depth for each pixel on the given images to
distinguish objects nearer to the camera from those farther away. The process is known as dense
depth triangulation and the output is known as the disparity map for a particular image. Due to
the different number and depth of objects present in the given images, we show the results for the
following two different views.

● Please note: The camera parameters have been recalculated for reusability of the code on
a new set of images as well.

Running the code:

Libraries imported: ​ numpy, cv2, glob, argparse, re

After setting the project_2a folder as the root directory, run the following command with suitable
arguments:
(An example has been provided for reference for left_5.png and right_5.png)

python3 ./code/task_4/task4.py --images_dir ./images/task_1/ --l_prefix left --r_prefix right
--undistort_rectified_img_l
./images/task_3_and_4/left_5.png
--undistort_rectified_img_r
./images/task_3_and_4/right_5.png --save_images_dir ./output/task_4/
