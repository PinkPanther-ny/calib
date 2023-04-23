import argparse
import yaml
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from core.calibration import Calib
from core.geometry import Vector3


def load_config(config_path: str) -> Dict:
    """
    Load the configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config: Dict) -> None:
    """
    Run the main program loop with the given configuration.
    """
    calib = Calib(filename=config["calib_filename"])
    cam = Camera(calib.width, calib.height, calib.new_camera_matrix, config["camera_position"], config["towards_direction"])
    # Plot the virtual screen
    cam.plot_virtual_screen()
    
class Camera:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_matrix: np.ndarray,
        camera_position: List[float],
        towards_direction: List[float],
    ):
        """
        Initialize the Camera class.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.camera_matrix = camera_matrix
        self.camera_position = Vector3(*camera_position)

        # Calculate aspect ratio from FOV
        self.h_fov_rad, self.v_fov_rad = self._calculate_field_of_view(degree=False)
        self.aspect_ratio = np.tan(self.h_fov_rad / 2) / np.tan(self.v_fov_rad / 2)
        self.camera_up, self.camera_right, self.camera_forward = self._camera_vectors(Vector3(*towards_direction))

        # Calculate the screen width and height from camera basis
        self.screen_height = 2 * np.tan(self.v_fov_rad / 2)
        self.screen_width = 2 * np.tan(self.h_fov_rad / 2)

        # Calculate the screen's center position in world coordinates
        self.screen_center = self.camera_position + self.camera_forward

        # Calculate the horizontal and vertical step size for each pixel
        self.h_step = self.screen_width / self.frame_width
        self.v_step = self.screen_height / self.frame_height

    def _camera_vectors(self, look_at: Vector3) -> tuple[Vector3, Vector3, Vector3]:
        world_up = Vector3(0, 1, 0)

        forward = look_at.normalized()
        right = np.cross(forward, world_up).view(Vector3).normalized()
        up = np.cross(right, forward).view(Vector3).normalized()

        return up, right, forward

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

    def get_pixel(self, x_pixel: int, y_pixel: int) -> Vector3:
        # Calculate the world coordinate for a specific pixel (i, j)
        x_offset = (x_pixel + 0.5) * self.h_step - self.screen_width / 2
        y_offset = (y_pixel + 0.5) * self.v_step - self.screen_height / 2

        pixel_position = self.screen_center + (self.camera_right * x_offset) - (self.camera_up * y_offset)
        return pixel_position

    def get_ray_direction(self, x_pixel: int, y_pixel: int) -> Vector3:
        ray_direction = self.get_pixel(x_pixel, y_pixel) - self.camera_position
        return ray_direction.normalized()

    def plot_virtual_screen(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(0, self.frame_width, 100):
            for j in range(0, self.frame_height, 100):
                pixel = self.get_pixel(i, j)
                ax.scatter(pixel.x, pixel.y, pixel.z, c='k', marker='.')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Virtual Screen in 3D Space')
        # Set the view angle (azimuth=0, elevation=90)
        ax.view_init(elev=90, azim=0, roll=90)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
    parser.add_argument('-c', '--config_path', type=str, default="configs/cam_deepblue.yaml", help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
