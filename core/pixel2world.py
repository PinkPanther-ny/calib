import cv2
import time
import numpy as np
from core.geometry import Plane, Ray, Vector3
from typing import Tuple, List


class CoordinateConverter:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        camera_position: List[float],
        towards_direction: List[float],
        pixel_is_distort: bool = True
    ):
        """
        Initialize the CoordinateConverter class.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_position = Vector3(*camera_position)
        self.towards_direction = Vector3(*towards_direction)
        self.ground_plane = Plane(Vector3(0, 0, 0), Vector3(0, 1, 0))
        
        self.pixel_is_distort = pixel_is_distort

        t0 = time.time()
        self.world_coordinate_map = np.empty((frame_height, frame_width, 3), dtype=np.float32)
        for i in range(frame_height):
            for j in range(frame_width):
                world_coord = self._pixel_to_world_coordinate((j, i))
                self.world_coordinate_map[i, j] = [world_coord.x, world_coord.y, world_coord.z]
        print(f"Coordinate map ({frame_width}, {frame_height}) computed in {time.time() - t0} secs.")

    def _calculate_fov(self) -> Tuple[float, float]:
        """
        Calculate the camera's field of view (FOV) in the x and y directions.
        """
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        w, h = self.camera_matrix[0, 2] * 2, self.camera_matrix[1, 2] * 2
        fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
        return fov_x, fov_y

    def pixel_to_world_coordinate(self, pixel: Tuple[float, float]) -> Vector3:
        """
        Convert a pixel coordinate to a world coordinate using the precomputed map.
        """
        x, y = pixel
        return Vector3(*self.world_coordinate_map[y, x])

    def _pixel_to_world_coordinate(
        self, pixel: Tuple[float, float]
    ) -> Vector3:
        """
        Convert a pixel coordinate to a world coordinate.
        """
        if self.pixel_is_distort:
            distorted_pixels = np.array([[pixel]], dtype=np.float32)
            undistorted_pixels = cv2.undistortPoints(distorted_pixels, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
            pixel = tuple(undistorted_pixels[0][0])

        fov_x, fov_y = self._calculate_fov()
        aspect_ratio = self.frame_width / self.frame_height

        x = (2 * (pixel[0] + 0.5) / self.frame_width - 1) * np.tan(fov_x / 2 * np.pi / 180) * aspect_ratio
        y = (1 - 2 * (pixel[1] + 0.5) / self.frame_height) * np.tan(fov_y / 2 * np.pi / 180)
        pixel_world_direction = Vector3(x, y, 1).normalized()

        rotation_axis = self.towards_direction.normalized().cross(Vector3(0, 0, 1))
        rotation_angle = np.arccos(self.towards_direction.normalized().dot(Vector3(0, 0, 1)))
        rotation_matrix = cv2.Rodrigues(np.array([rotation_axis.x, rotation_axis.y, rotation_axis.z]) * rotation_angle)[0]
        rotated_direction = np.dot(rotation_matrix, np.array([pixel_world_direction.x, pixel_world_direction.y, pixel_world_direction.z]))

        rotated_pixel_world_direction = Vector3(rotated_direction[0], rotated_direction[1], rotated_direction[2]).normalized()

        ray = Ray(self.camera_position, rotated_pixel_world_direction)
        intersection = self.ground_plane.intersect(ray)

        if intersection is None:
            return Vector3(float('inf'), float('inf'), float('inf'))
        else:
            return intersection
