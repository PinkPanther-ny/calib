import argparse
import yaml
import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import List
from core import Calib
from glutil import GLGrid, GLCompass, draw_text

class Renderer:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        camera_position: List[float],
        towards_direction: List[float],
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_position = camera_position
        self.towards_direction = np.array(towards_direction)
        self.setup()
        self.grid = GLGrid(20, 0.5)
        self.compass = GLCompass(self.frame_width, self.frame_height)

    def setup(self):
        # Initialize Pygame, pygame.OPENGL indicates that the window should be created with an OpenGL context.
        pygame.init()

        # Set up multisampling
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 8)
        pygame.display.set_mode((self.frame_width, self.frame_height), pygame.DOUBLEBUF | pygame.OPENGL)

        # Enable multisampling in the OpenGL context
        glEnable(GL_MULTISAMPLE)
        
        # Set up OpenGL
        glClearColor(0.5, 0.5, 0.5, 1)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.frame_width / self.frame_height), 0.1, 100.0)
        glEnable(GL_DEPTH_TEST)
        
        # Set up the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera_target = [self.camera_position[i] + self.towards_direction[i] for i in range(3)]
        gluLookAt(*self.camera_position, *camera_target, 0, 1, 0)

        # Set up the viewport
        glViewport(0, 0, self.frame_width, self.frame_height)

    def render(self):
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        camera_target = [self.camera_position[i] + self.towards_direction[i] for i in range(3)]
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.camera_position, *camera_target, 0, 1, 0)
        
        self.grid.draw(colored=True)
        self.compass.draw(self.towards_direction)
        draw_text(f"Camera Position: {np.round(self.camera_position, 2)}", 10, 30, self.frame_width, self.frame_height)
        draw_text(f"Towards Direction: {np.round(self.towards_direction, 2)}", 10, 10, self.frame_width, self.frame_height)
    
    def run(self):
        clock = pygame.time.Clock()

        lifting_speed = 0.05
        move_speed = 0.02
        mouse_sensitivity = 0.2

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                                    # Mouse scroll event
                if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 4:  # Scroll up
                        self.camera_position[1] += lifting_speed
                    elif event.button == 5:  # Scroll down
                        self.camera_position[1] -= lifting_speed

            keys = pygame.key.get_pressed()

            if keys[pygame.K_ESCAPE]:
                running = False
                continue
            
            # Calculate movement vector
            move_vector = [0.0, 0.0, 0.0]
            if keys[pygame.K_w]:
                move_vector[0] += self.towards_direction[0]
                move_vector[2] += self.towards_direction[2]
            if keys[pygame.K_s]:
                move_vector[0] -= self.towards_direction[0]
                move_vector[2] -= self.towards_direction[2]
            if keys[pygame.K_d]:
                move_vector[0] -= self.towards_direction[2]
                move_vector[2] += self.towards_direction[0]
            if keys[pygame.K_a]:
                move_vector[0] += self.towards_direction[2]
                move_vector[2] -= self.towards_direction[0]

            # Normalize movement vector
            move_vector_length = np.linalg.norm(move_vector)
            if move_vector_length > 1e-6:
                move_vector_normalized = np.array([coord / move_vector_length for coord in move_vector])
            else:
                move_vector_normalized = np.array(move_vector)

            if keys[pygame.K_LSHIFT]:
                move_vector_normalized *= 3

            # Update camera position with normalized movement vector
            self.camera_position[0] += move_speed * move_vector_normalized[0]
            self.camera_position[2] += move_speed * move_vector_normalized[2]

            # Mouse movement
            pygame.event.get_grab()
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            mouse_dx, mouse_dy = pygame.mouse.get_rel()

            yaw = mouse_dx * mouse_sensitivity
            pitch = -mouse_dy * mouse_sensitivity

            x, y, z = self.towards_direction
            xz_len = np.sqrt(x * x + z * z)
            xz_len_new = xz_len * np.cos(np.radians(pitch)) - y * np.sin(np.radians(pitch))
            y_new = xz_len * np.sin(np.radians(pitch)) + y * np.cos(np.radians(pitch))

            if xz_len_new > 1e-6:
                x_new = x * (xz_len_new / xz_len) - z * np.tan(np.radians(yaw))
                z_new = z * (xz_len_new / xz_len) + x * np.tan(np.radians(yaw))

                self.towards_direction = np.array([x_new, y_new, z_new])
                self.towards_direction = self.towards_direction / np.linalg.norm(self.towards_direction)

            self.render()

            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


def load_config(config_path: str):
    """
    Load the configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
    parser.add_argument('-c', '--config_path', type=str, default="configs/cam_deepblue.yaml", help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    calib = Calib(filename=config["calib_filename"])
    camera_matrix, dist_coeffs = calib.camera_matrix, calib.dist_coeffs
    frame_width, frame_height = calib.width, calib.height

    camera_position = config["camera_position"]
    towards_direction = config["towards_direction"]
    
    renderer = Renderer(
        frame_width, frame_height, 
        camera_matrix, dist_coeffs,
        camera_position, towards_direction
    )
    
    renderer.run()
    # grid_image = renderer.image
    # cv2.imwrite("im.png", grid_image)