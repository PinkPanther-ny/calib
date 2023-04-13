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
        self.towards_direction = Vector3(*towards_direction).normalized()
        self.ground_plane = Plane(Vector3(0, 0, 0), Vector3(0, 1, 0))
        
        self.pixel_is_distort = pixel_is_distort

        t0 = time.time()
        self._create_world_coordinate_map_fast()
        print(f"Coordinate map ({frame_width}, {frame_height}) computed in {time.time() - t0} secs.")
        
        self.world_coordinate_map_to_colored_depth_image(max_depth=3)

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
            undistorted_pixels = cv2.undistortPointsIter(distorted_pixels, self.camera_matrix, self.dist_coeffs, R=None, P=self.camera_matrix, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.03))
            pixel = tuple(undistorted_pixels[0][0])

        fov_x, fov_y = self._calculate_fov()
        aspect_ratio = self.frame_width / self.frame_height

        x = (2 * (pixel[0] + 0.5) / self.frame_width - 1) * np.tan(fov_x / 2 * np.pi / 180) * aspect_ratio
        y = (1 - 2 * (pixel[1] + 0.5) / self.frame_height) * np.tan(fov_y / 2 * np.pi / 180)
        pixel_world_direction = Vector3(x, y, 1).normalized()

        rotation_axis = np.cross(self.towards_direction, Vector3(0, 0, 1))
        rotation_angle = np.arccos(np.dot(self.towards_direction, Vector3(0, 0, 1)))
        rotation_matrix = cv2.Rodrigues(rotation_axis * rotation_angle)[0]
        rotated_direction = np.dot(rotation_matrix, pixel_world_direction)

        rotated_pixel_world_direction = Vector3(*rotated_direction).normalized()

        ray = Ray(self.camera_position, rotated_pixel_world_direction)
        intersection = self.ground_plane.intersect(ray)

        if intersection is None:
            return Vector3(float('inf'), float('inf'), float('inf'))
        else:
            return intersection

    def _create_world_coordinate_map(self):
        self.world_coordinate_map = np.empty((self.frame_height, self.frame_width, 3), dtype=np.float32)
        for i in range(self.frame_height):
            for j in range(self.frame_width):
                world_coord = self._pixel_to_world_coordinate((j, i))
                self.world_coordinate_map[i, j] = [world_coord.x, world_coord.y, world_coord.z]

    def _create_world_coordinate_map_fast(self):
        self.world_coordinate_map = np.empty((self.frame_height, self.frame_width, 3), dtype=np.float32)
        
        pixel_x, pixel_y = np.meshgrid(np.arange(self.frame_width, dtype=np.float32), np.arange(self.frame_height, dtype=np.float32))
        pixel_coords = np.vstack((pixel_x.flatten(), pixel_y.flatten())).T
        
        if self.pixel_is_distort:
            undistorted_pixel_coords = cv2.undistortPointsIter(pixel_coords.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs, R=None, P=self.camera_matrix, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.03))
            pixel_coords = undistorted_pixel_coords.reshape(-1, 2)

        fov_x, fov_y = self._calculate_fov()
        aspect_ratio = self.frame_width / self.frame_height
        
        x = (2 * (pixel_coords[:, 0] + 0.5) / self.frame_width - 1) * np.tan(fov_x / 2 * np.pi / 180) * aspect_ratio
        y = (1 - 2 * (pixel_coords[:, 1] + 0.5) / self.frame_height) * np.tan(fov_y / 2 * np.pi / 180)
        pixel_world_direction = np.vstack((x, y, np.ones_like(x)))
        pixel_world_direction = (pixel_world_direction / np.linalg.norm(pixel_world_direction.T, axis=1)).T

        rotation_axis = np.cross(self.towards_direction, Vector3(0, 0, 1))
        rotation_angle = np.arccos(self.towards_direction.dot(Vector3(0, 0, 1)))
        rotation_matrix = cv2.Rodrigues(rotation_axis * rotation_angle)[0]
        rotated_direction = np.dot(pixel_world_direction, rotation_matrix.T)

        rotated_pixel_world_direction = (rotated_direction.T / np.linalg.norm(rotated_direction, axis=1)).T

        intersection = self.ground_plane.intersect_many(
            ray_origin=self.camera_position, 
            ray_directions=np.array(rotated_pixel_world_direction)
        )

        self.world_coordinate_map = intersection.reshape(self.frame_height, self.frame_width, 3)

    def world_coordinate_map_to_colored_depth_image(self, max_depth: float = 100.0) -> np.ndarray:
        """
        Convert a world coordinate map to a colored depth image.
        
        Args:
            world_coordinate_map (np.ndarray): A 3D numpy array containing world coordinates (x, y, z) for each pixel.
            max_depth (float): The maximum depth value to be visualized, used for scaling the depth values.
            
        Returns:
            np.ndarray: A colored depth image.
        """
        # Extract the depth (z) values from the world_coordinate_map
        depth_map = self.world_coordinate_map[:, :, 2]

        # Create a mask for infinite depth values
        inf_depth_mask = np.isinf(depth_map)

        # Normalize the depth values to the range [0, 1] based on the maximum depth value
        depth_map_normalized = np.clip(depth_map / max_depth, 0, 1)

        # Convert the normalized depth values to a grayscale image
        gray_depth_image = (depth_map_normalized * 255).astype(np.uint8)

        # Apply a colormap to the grayscale depth image to create a colored depth image
        colored_depth_image = cv2.applyColorMap(gray_depth_image, cv2.COLORMAP_JET)

        # Set the color of infinite depth values to white
        colored_depth_image[inf_depth_mask] = [128, 128, 128]

        cv2.imwrite("color.png", colored_depth_image)
        return colored_depth_image
