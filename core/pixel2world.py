from threading import Thread
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
        camera_position: List[float],
        towards_direction: List[float],
        auto_update: bool,
    ):
        """
        Initialize the CoordinateConverter class.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_matrix = camera_matrix
        self.camera_position = Vector3(*camera_position)
        self.towards_direction = Vector3(*towards_direction).normalized()
        self.ground_plane = Plane(Vector3(0, 0, 0), Vector3(0, 1, 0))
        
        t0 = time.time()
        self._generate_world_coordinates_map_fast()
        print(f"Coordinate map ({frame_width}, {frame_height}) computed in {time.time() - t0} secs.")
        
        self.generate_depth_map(max_depth=2.5)
        
        if auto_update:
            self.update_thread = Thread(target=self._auto_update_depth_map)
            self.update_thread.daemon = True
            self.update_thread.start()

    def _calculate_field_of_view(self, degree=True) -> Tuple[float, float]:
        """
        Calculate the camera's field of view (FOV) in the x and y directions.
        """
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        fov_x = 2 * np.arctan(self.frame_width / (2 * fx))
        fov_y = 2 * np.arctan(self.frame_height / (2 * fy))
        if degree:
            return fov_x * 180 / np.pi, fov_y * 180 / np.pi
        else:
            return fov_x, fov_y

    def pixel_to_world(self, pixel: Tuple[float, float]) -> Vector3:
        """
        Convert a pixel coordinate to a world coordinate using the precomputed map.
        """
        x, y = pixel
        return Vector3(*self.world_coordinates_map[y, x])

    def _pixel_to_world(
        self, pixel: Tuple[float, float]
    ) -> Vector3:
        """
        Convert a pixel coordinate to a world coordinate.
        """

        fov_x, fov_y = self._calculate_field_of_view(degree=False)
        aspect_ratio = self.frame_width / self.frame_height

        # Calculate the camera's coordinate system
        up = np.array([0, 1, 0])
        horizontal = np.cross(self.towards_direction, up)
        vertical = np.cross(horizontal, self.towards_direction)

        # Find the center of the image plane in world coordinates
        image_plane_center = self.camera_position + self.towards_direction

        pixel_x, pixel_y = pixel
        # Calculate the position of the pixel in the world coordinate system
        pixel_u = (2 * (pixel_x + 0.5) / self.frame_width - 1) * np.tan(fov_x / 2) * aspect_ratio
        pixel_v = (1 - 2 * (pixel_y + 0.5) / self.frame_height) * np.tan(fov_y / 2)
        pixel_position_in_world = image_plane_center + pixel_u * horizontal + pixel_v * vertical

        # Create the ray direction vector and normalize it
        ray_direction = pixel_position_in_world - self.camera_position

        ray = Ray(self.camera_position, ray_direction)
        intersection = self.ground_plane.intersect(ray)

        if intersection is None:
            return Vector3(float('inf'), float('inf'), float('inf'))
        else:
            return intersection

    def _generate_world_coordinates_map(self):
        self.world_coordinates_map = np.empty((self.frame_height, self.frame_width, 3), dtype=np.float32)
        for i in range(self.frame_height):
            for j in range(self.frame_width):
                world_coord = self._pixel_to_world((j, i))
                self.world_coordinates_map[i, j] = [world_coord.x, world_coord.y, world_coord.z]

    def _generate_world_coordinates_map_fast(self):
        # Create a grid of pixel coordinates
        pixel_x, pixel_y = np.meshgrid(np.arange(self.frame_width), np.arange(self.frame_height))

        # Calculate the u and v values for each pixel
        fov_x, fov_y = self._calculate_field_of_view(degree=False)
        aspect_ratio = self.frame_width / self.frame_height
        pixel_u = (2 * (pixel_x + 0.5) / self.frame_width - 1) * np.tan(fov_x / 2) * aspect_ratio
        pixel_v = (1 - 2 * (pixel_y + 0.5) / self.frame_height) * np.tan(fov_y / 2)

        # Calculate the camera's coordinate system
        up = np.array([0, 1, 0])
        horizontal = np.cross(self.towards_direction, up)
        vertical = np.cross(horizontal, self.towards_direction)

        # Find the center of the image plane in world coordinates
        image_plane_center = self.camera_position + self.towards_direction

        # Calculate the position of each pixel in the world coordinate system
        pixel_positions_in_world = image_plane_center + pixel_u[..., np.newaxis] * horizontal + pixel_v[..., np.newaxis] * vertical

        # Create the ray direction vectors and normalize them
        ray_directions = pixel_positions_in_world - self.camera_position

        # Calculate the intersection points of the rays with the ground plane
        intersections = self.ground_plane.intersect_many(self.camera_position, ray_directions)

        # Create the world_coordinates_map from the intersections
        self.world_coordinates_map = intersections.astype(np.float32)

    def generate_depth_map(self, max_depth: float = 100.0, grid_spacing: float = 0.5) -> np.ndarray:
        """
        Convert a world coordinate map to a colored depth image with a grid.
        
        Args:
            max_depth (float): The maximum depth value to be visualized, used for scaling the depth values.
            grid_spacing (float): The spacing between grid lines in the world coordinate system.
                
        Returns:
            np.ndarray: A colored depth image with a grid.
        """
        # Extract the depth (z) values from the world_coordinates_map
        depth_map = self.world_coordinates_map[:, :, 2]

        # Create a mask for infinite depth values
        inf_depth_mask = np.isinf(depth_map)

        # Normalize the depth values to the range [0, 1] based on the maximum depth value
        depth_map_normalized = np.clip(depth_map / max_depth, 0, 1)

        # Convert the normalized depth values to a grayscale image
        gray_depth_image = (depth_map_normalized * 255).astype(np.uint8)

        # Apply a colormap to the grayscale depth image to create a colored depth image
        depth_map = cv2.applyColorMap(gray_depth_image, cv2.COLORMAP_JET)

        # Set the color of infinite depth values to white
        depth_map[inf_depth_mask] = [128, 128, 128]

        # Draw grid lines
        x_coords = self.world_coordinates_map[:, :, 0]
        z_coords = self.world_coordinates_map[:, :, 2]
        
        x_on_grid = np.abs(x_coords % grid_spacing) < 1e-2
        z_on_grid = np.abs(z_coords % grid_spacing) < 1e-2
        grid_mask = np.logical_or(x_on_grid, z_on_grid)
        
        depth_map[grid_mask] = [0, 0, 0]

        self.depth_map = depth_map

    def update_depth_map(self, max_depth: float = 2.5):
        """
        Recalculate the world_coordinates_map and update the depth_map.

        Args:
            max_depth (float): The maximum depth value to be visualized, used for scaling the depth values.
        """
        self.towards_direction = Vector3(*self.towards_direction).normalized()
        self._generate_world_coordinates_map_fast()
        self.generate_depth_map(max_depth=max_depth)

    def _auto_update_depth_map(self):
        while True:
            self.update_depth_map()
