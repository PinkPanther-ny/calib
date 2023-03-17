from pixel2world import Vector3, pixel_to_world_coordinate
from calibration import Calib
import cv2


# Load calibration data
calib = Calib(filename="calib_data.json")
camera_matrix, dist_coeffs = calib.camera_matrix, calib.dist_coeffs

# Define example parameters
frame_width = 640
frame_height = 480
camera_position = Vector3(0, 0.75, 0)
towards_direction = Vector3(0, 0, 1)


def on_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        pixel = (x, y)
        intersection = pixel_to_world_coordinate(pixel, frame_width, frame_height, camera_matrix, camera_position, towards_direction)
        print(f"Intersection point: x={intersection.x}, y={intersection.y}, z={intersection.z}")

# Open camera
cap = cv2.VideoCapture(1)
cv2.namedWindow("Camera Frame")
cv2.setMouseCallback("Camera Frame", on_mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = calib.undistort(frame, crop=False)
    cv2.imshow("Camera Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
