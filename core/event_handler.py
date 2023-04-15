import cv2
import numpy as np
from typing import List
from core.pixel2world import CoordinateConverter


class MouseEventHandler:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_matrix: np.ndarray,
        camera_position: List[float],
        towards_direction: List[float],
    ):
        """
        Initialize the MouseEventHandler class.
        """
        self.coordinate_converter = CoordinateConverter(
            frame_width, frame_height, camera_matrix, camera_position, towards_direction
        )
        self.text_to_display = ""
        self.cursor_position = (0, 0)

    def on_mouse_event(
        self, event: int, x: int, y: int, flags: int, param: None
    ) -> None:
        """
        Handle mouse events and update the intersection point.
        """
        self.cursor_position = (x, y)
        intersection = self.coordinate_converter.pixel_to_world_coordinate(
            self.cursor_position
        )
        # intersection[2] = intersection[2] / 1.09

        self.text_to_display = str(intersection)
        print(f"Intersection point: {intersection}")

    def display_text_on_frame(self, frame: np.ndarray) -> None:
        """
        Display the intersection point text on the frame.
        """
        cv2.putText(frame, self.text_to_display, self.cursor_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 255, 128), 1)
