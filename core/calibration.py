import cv2
import glob
import json
import numpy as np
from typing import Optional, Tuple, Union


class Calib:
    def __init__(
        self,
        image_dir: str = "./images/*.jpg",
        chessboard_size: Tuple[int, int] = (8, 12),
        filename: Optional[str] = None,
    ):
        """
        Initialize the Calib class, either by loading existing calibration data or by calibrating the camera.
        """
        if filename is not None:
            self.load(filename)
            return

        # Chessboard dimensions is umber of internal corners (width, height)
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        # Read a set of calibration images
        calibration_images = glob.glob(image_dir)  # Replace with your images path

        first_image = True
        for fname in calibration_images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if first_image:
                self.width, self.height = gray.shape[::-1]
                first_image = False
            else:
                if gray.shape[::-1] != (self.width, self.height):
                    raise ValueError("All images must have the same dimensions for calibration.")

            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            # If found, add object points and image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)

        # Calibrate the camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def save(self, filename: str = "calib_data.json") -> None:
        """
        Save the calibration data to a JSON file.
        """
        calibration_data = {
            "width": self.width,
            "height": self.height,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
        }

        with open(filename, "w") as f:
            json.dump(calibration_data, f, indent=4)

    def load(self, filename: str = "calib_data.json") -> dict:
        """
        Load the calibration data from a JSON file.
        """
        with open(filename, "r") as f:
            calibration_data = json.load(f)

        self.width = calibration_data["width"]
        self.height = calibration_data["height"]
        self.camera_matrix = np.array(calibration_data["camera_matrix"])
        self.dist_coeffs = np.array(calibration_data["dist_coeffs"])

        return {
            "width": self.width,
            "height": self.height,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
        }

    def undistort(
        self,
        image: Union[str, np.ndarray],
        output: Optional[str] = None,
        crop: bool = False,
    ) -> np.ndarray:
        """
        Undistort an image using the calibration data.
        """
        if type(image)==str:
            image = cv2.imread(image)

        h,  w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, 
            self.dist_coeffs, 
            (w, h), 
            0
        )

        # undistort
        image = cv2.undistort(
            image, 
            self.camera_matrix, 
            self.dist_coeffs, 
            None, 
            newcameramtx
        )

        if crop:
            # crop the image
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]

        if output is not None:
            cv2.imwrite(output, image)

        return image

    def distort(self, image: Union[str, np.ndarray], output: Optional[str] = None) -> np.ndarray:
        """
        Distort an image using the calibration data.
        """
        if type(image) == str:
            image = cv2.imread(image)

        h, w = image.shape[:2]

        # Compute the distortion maps for undistortion
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (w, h),
            cv2.CV_32FC1,
        )

        # Undistort the image
        undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

        # Compute the distortion maps for distortion using the inverse of the distortion coefficients
        inv_dist_coeffs = -1 * self.dist_coeffs
        map1_inv, map2_inv = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            inv_dist_coeffs,
            None,
            self.camera_matrix,
            (w, h),
            cv2.CV_32FC1,
        )

        # Distort the image
        distorted_image = cv2.remap(undistorted_image, map1_inv, map2_inv, cv2.INTER_LINEAR)

        if output is not None:
            cv2.imwrite(output, distorted_image)

        return distorted_image

    def __repr__(self) -> str:
        """
        Represent the Calib object as a string.
        """
        return json.dumps({
            "width": self.width,
            "height": self.height,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
        }, indent=4)


if __name__ == "__main__":
    cam1 = False
    if cam1:
        calib = Calib(image_dir="fisheye/*.jpg", chessboard_size=(7, 12))
        calib.save(filename="configs/calib_fish.json")
        calib.undistort("fisheye/WIN_20230323_14_10_45_Pro.jpg", "calibrated.jpg", crop=False)
        calib.load(filename="configs/calib_fish.json")
        print(calib)
    else:
        calib = Calib(image_dir="images/*.jpg", chessboard_size=(7, 12))
        calib.save(filename="configs/calib_data.json")
        calib.undistort("images/WIN_20230317_19_15_37_Pro.jpg", "calibrated.jpg", crop=False)
        calib.distort("calibrated.jpg", "dis.jpg")
        calib.load(filename="configs/calib_data.json")
        print(calib)
