from threading import Thread
import cv2
import time
import numpy as np
from core.geometry import Plane, Vector3
from typing import Tuple


class CoordinateConverter:
    def __init__(
        self,
        camera,
        frame_width: int,
        frame_height: int,
        auto_update: bool,
    ):
        """
        Initialize the CoordinateConverter class.
        """
        self.depth_map = None
        self.camera = camera
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ground_plane = Plane(Vector3(0, 0, 0), Vector3(0, 1, 0))
        
        t0 = time.time()
        self._generate_world_coordinates_map_fast()
        print(f"Coordinate map ({frame_width}, {frame_height}) computed in {time.time() - t0} secs.")
        
        self.generate_depth_map(max_depth=2.5)
        
        if auto_update:
            self.update_thread = Thread(target=self._auto_update_depth_map)
            self.update_thread.daemon = True
            self.update_thread.start()

    def pixel_to_world(self, pixel: Tuple[float, float]) -> Vector3:
        """
        Convert a pixel coordinate to a world coordinate using the precomputed map.
        """
        x, y = pixel
        return Vector3(*self.world_coordinates_map[y, x])

    def _generate_world_coordinates_map(self):
        self.world_coordinates_map = np.empty((self.frame_height, self.frame_width, 3), dtype=np.float32)
        for pixel_y in range(self.frame_height):
            for pixel_x in range(self.frame_width):

                intersection = self.ground_plane.intersect(self.camera.get_ray_at_pixel(pixel_x, pixel_y))
                self.world_coordinates_map[pixel_y, pixel_x] = [intersection.x, intersection.y, intersection.z]

    def _generate_world_coordinates_map_fast(self):
        # Calculate the intersection points of the rays with the ground plane
        intersections = self.ground_plane.intersect_many(*self.camera.get_all_ray_directions())

        # Create the world_coordinates_map from the intersections
        self.world_coordinates_map = intersections.astype(np.float32)

    def generate_depth_map(self, max_depth: float = 100.0, grid_spacing: float = 0.5) -> None:
        """
        Convert a world coordinate map to a colored depth image with a grid.
        
        Args:
            max_depth (float): The maximum depth value to be visualized, used for scaling the depth values.
            grid_spacing (float): The spacing between grid lines in the world coordinate system.
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
        self._generate_world_coordinates_map_fast()
        self.generate_depth_map(max_depth=max_depth)

    def _auto_update_depth_map(self):
        while True:
            self.update_depth_map()
