#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# Extracting path of individual image stored in a given directory
images = glob.glob('./images_s/*.jpg')

# Defining the dimensions of checkerboard
CHECKERBOARD = (8,12)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 0.75
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
text = "ESC or Q to close window"

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

imgs = []
for fname in tqdm(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    imgs.append(img)    
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\n\n\n")
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# dist[0][3]=0
# dist[0][4]=0
for img in tqdm(imgs):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # # undistort
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    
    # Using cv2.putText() method
    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('img', img)
    key = chr(cv2.waitKey(0)%256)
    if key == '\x1b' or key == 'q':
        break
    
    # Using cv2.putText() method
    dst = cv2.putText(dst, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('img', dst)
    key = repr(chr(cv2.waitKey(0)%256))
    if key == '\x1b' or key == 'q':
        break

cv2.destroyAllWindows()

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )