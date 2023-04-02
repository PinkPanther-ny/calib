import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import pygame
from typing import List

from core import Calib
import argparse
import yaml

def load_config(config_path: str):
    """
    Load the configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class Renderer:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_matrix: np.ndarray,
        camera_position: List[float],
        towards_direction: List[float],
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_matrix = camera_matrix
        self.camera_position = camera_position
        self.towards_direction = towards_direction
        
        # Initialize Pygame, pygame.OPENGL indicates that the window should be created with an OpenGL context.
        pygame.init()
        pygame.display.set_mode((self.frame_width, self.frame_height), pygame.DOUBLEBUF | pygame.OPENGL)
        self.compute()

    def draw_grid(self, size, spacing):
        glBegin(GL_LINES)
        for i in range(-size, size + 1):
            glVertex3f(i * spacing, 0, -size * spacing)
            glVertex3f(i * spacing, 0, size * spacing)
            glVertex3f(-size * spacing, 0, i * spacing)
            glVertex3f(size * spacing, 0, i * spacing)
        glEnd()

    def compute(self):
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

        # Draw the grid
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(0, 0, 0)
        self.draw_grid(100, 0.5)

    def run(self):
        clock = pygame.time.Clock()

        move_speed = 0.05
        mouse_sensitivity = 0.2

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.camera_position[0] += move_speed * self.towards_direction[0]
                self.camera_position[2] += move_speed * self.towards_direction[2]
            if keys[pygame.K_s]:
                self.camera_position[0] -= move_speed * self.towards_direction[0]
                self.camera_position[2] -= move_speed * self.towards_direction[2]
            if keys[pygame.K_d]:
                self.camera_position[0] -= move_speed * self.towards_direction[2]
                self.camera_position[2] += move_speed * self.towards_direction[0]
            if keys[pygame.K_a]:
                self.camera_position[0] += move_speed * self.towards_direction[2]
                self.camera_position[2] -= move_speed * self.towards_direction[0]

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

                self.towards_direction = [x_new, y_new, z_new]

            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set up the camera
            camera_target = [self.camera_position[i] + self.towards_direction[i] for i in range(3)]
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(*self.camera_position, *camera_target, 0, 1, 0)

            # Draw the grid
            glColor3f(0, 0, 0)
            self.draw_grid(100, 0.5)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()



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
        camera_matrix,
        camera_position, towards_direction
    )
    
    renderer.run()
    # grid_image = renderer.image
    # cv2.imwrite("im.png", grid_image)