import cv2
import numpy as np
from typing import List
from core.pixel2world import CoordinateConverter


class MouseEventHandler:
    def __init__(
        self,
        converter: CoordinateConverter
    ):
        """
        Initialize the MouseEventHandler class.
        """
        self.converter = converter
        self._text_to_display = ""
        self._cursor_position = (0, 0)

    def on_mouse_event(
        self, event: int, x: int, y: int, flags: int, param: None
    ) -> None:
        """
        Handle mouse events and update the intersection point.
        """
        self._cursor_position = (x, y)
        intersection = self.converter.pixel2world(
            self._cursor_position
        )

        self._text_to_display = str(intersection)
        print(f"Intersection point: {intersection}")

    def display_text_on_frame(self, frame: np.ndarray) -> None:
        """
        Display the intersection point text on the frame.
        """
        cv2.putText(frame, self._text_to_display, 
                    self._cursor_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, (128, 255, 128), 1)
