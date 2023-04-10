# Pixel2World

Pixel2World is a project that allows you to convert pixel coordinates in an image to real-world coordinates using camera calibration and a mouse event handler.

## Dependencies

- OpenCV
- PyYAML
- typing

## Usage

1. Install the required dependencies.
2. Modify the `config_path` variable in the main.py file or use the `-c` argument to specify a custom configuration file.
3. Run the main.py script.

```
bash
python main.py
python main.py -c /path/to/your/config.yaml
```

## Configuration

The configuration file (YAML format) should include the following fields:

- calib_filename: Path to the camera calibration file.
- camera_position: Position of the camera in the real world.
- towards_direction: Direction the camera is pointing towards.
- camera_index: Index of the camera device.

## Code Structure
The main components of the project are:

- `Calib`: A class to handle camera calibration.
- `MouseEventHandler`: A class to handle mouse events and coordinate conversion.
- `Renderer`: A class to render the grid and undistorted image (commented out in main.py).
The main loop of the application captures video frames from the camera, handles mouse events, and displays the processed image in a window. Press 'q' to exit the loop and close the application.