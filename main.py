import argparse
import cv2
import yaml
from typing import Dict
from core import Calib, MouseEventHandler, Gyroscope, combine_image


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
    camera_matrix, dist_coeffs = calib.new_camera_matrix, calib.dist_coeffs
    frame_width, frame_height = calib.width, calib.height

    camera_position = config["camera_position"]
    towards_direction = config["towards_direction"]
    
    try:
        gyro = Gyroscope("COM4")
        towards_direction = gyro.data["towards_direction"]
        print(f"Gyroscope detected, towards direction: {towards_direction}")
    except Gyroscope.GyroscopeIniFailed as e:
        print(str(e))
        print(f"Using towards direction from config: {towards_direction}")
    
    mouse_event_handler = MouseEventHandler(
        frame_width, frame_height, camera_matrix,
        camera_position, towards_direction
    )

    cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_DSHOW)
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cv2.namedWindow("Camera Frame")
    cv2.setMouseCallback("Camera Frame", mouse_event_handler.on_mouse_event)

    while cv2.getWindowProperty("Camera Frame", cv2.WND_PROP_VISIBLE) >= 1:
        ret, frame = cap.read()
        frame = calib.undistort(frame, crop=False)
        if not ret:
            break
        mouse_event_handler.display_text_on_frame(frame)
        cv2.imshow("Camera Frame", combine_image(frame, mouse_event_handler.coordinate_converter.colored_depth_image))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
    parser.add_argument('-c', '--config_path', type=str, default="configs/cam_deepblue.yaml", help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
