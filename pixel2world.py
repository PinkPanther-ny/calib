import numpy as np
import cv2


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def normalized(self):
        length = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector3(self.x / length, self.y / length, self.z / length)
    
    def __repr__(self) -> str:
        return f"(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalized()

class Plane:
    def __init__(self, center, normal):
        self.center = center
        self.normal = normal.normalized()

    def intersect(self, ray):
        if self.normal.dot(ray.direction) != 0:
            t = (self.center - ray.origin).dot(self.normal) / ray.direction.dot(self.normal)
            if t >= 0:
                intersection = ray.origin + ray.direction * t
                return intersection
        return None

def calculate_fov(camera_matrix):
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    w, h = camera_matrix[0, 2] * 2, camera_matrix[1, 2] * 2
    fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
    return fov_x, fov_y

def pixel_to_world_coordinate(pixel, 
                              frame_width, frame_height, 
                              camera_matrix, dist_coeffs, 
                              camera_position, towards_direction, 
                              ground_plane=Plane(Vector3(0, 0, 0), Vector3(0, 1, 0)), 
                              pixel_is_distort=True):
    if pixel_is_distort:
        # Undistort the pixel coordinates
        distorted_pixels = np.array([[pixel]], dtype=np.float32)
        undistorted_pixels = cv2.undistortPoints(distorted_pixels, camera_matrix, dist_coeffs, P=camera_matrix)
        pixel = tuple(undistorted_pixels[0][0])

    fov_x, fov_y = calculate_fov(camera_matrix)
    aspect_ratio = frame_width / frame_height

    x = (2 * (pixel[0] + 0.5) / frame_width - 1) * np.tan(fov_x / 2 * np.pi / 180) * aspect_ratio
    y = (1 - 2 * (pixel[1] + 0.5) / frame_height) * np.tan(fov_y / 2 * np.pi / 180)
    pixel_world_direction = Vector3(x, y, 1).normalized()

    # Rotate the pixel_world_direction according to the towards_direction
    rotation_axis = towards_direction.normalized().cross(Vector3(0, 0, 1))
    rotation_angle = np.arccos(towards_direction.normalized().dot(Vector3(0, 0, 1)))
    rotation_matrix = cv2.Rodrigues(np.array([rotation_axis.x, rotation_axis.y, rotation_axis.z]) * rotation_angle)[0]
    rotated_direction = np.dot(rotation_matrix, np.array([pixel_world_direction.x, pixel_world_direction.y, pixel_world_direction.z]))

    rotated_pixel_world_direction = Vector3(rotated_direction[0], rotated_direction[1], rotated_direction[2]).normalized()
    
    ray = Ray(camera_position, rotated_pixel_world_direction)
    intersection = ground_plane.intersect(ray)
    
    if intersection is None:
        return Vector3(float('inf'), float('inf'), float('inf'))
    else:
        return intersection


if __name__ == "__main__":
    from calibration import Calib
    
    # Load calibration data
    calib = Calib(filename="calib_data.json")
    camera_matrix, dist_coeffs = calib.camera_matrix, calib.dist_coeffs

    # Define example parameters
    frame_width = 640
    frame_height = 480
    camera_position = Vector3(0, 0.75, 0)
    towards_direction = Vector3(0, 0, 1)

    text_to_display = ""
    cursor_position = (0, 0)
    def on_mouse_event(event, x, y, flags, param):
        global text_to_display, cursor_position
        cursor_position = (x, y)
        intersection = pixel_to_world_coordinate(cursor_position, 
                                                    frame_width, 
                                                    frame_height, 
                                                    camera_matrix, 
                                                    dist_coeffs,
                                                    camera_position, 
                                                    towards_direction,
                                                    pixel_is_distort=True)
        
        text_to_display = str(intersection)
        print(f"Intersection point: {intersection}")

    # Open camera
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Camera Frame")
    cv2.setMouseCallback("Camera Frame", on_mouse_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Draw the text on the frame
        cv2.putText(frame, text_to_display, cursor_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 255, 128), 1)
        cv2.imshow("Camera Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
