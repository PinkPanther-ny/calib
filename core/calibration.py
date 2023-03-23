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

        for fname in calibration_images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
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

        self.camera_matrix = np.array(calibration_data["camera_matrix"])
        self.dist_coeffs = np.array(calibration_data["dist_coeffs"])

        return {
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

    def __repr__(self) -> str:
        """
        Represent the Calib object as a string.
        """
        return str({
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
        })


if __name__ == "__main__":

    calib = Calib(image_dir="./images/*.jpg", chessboard_size=(7, 12))
    calib.save(filename="calib_data.json")
    calib.undistort("./images/WIN_20230317_19_15_37_Pro.jpg", "calibrated.jpg", crop=False)
    calib.load(filename="calib_data.json")
    print(calib)
