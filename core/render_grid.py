import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import pygame
from typing import List


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
        self.towards_direction = towards_direction
        
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
        # Initialize Pygame, pygame.OPENGL indicates that the window should be created with an OpenGL context.
        pygame.init()
        pygame.display.set_mode((self.frame_width, self.frame_height), pygame.DOUBLEBUF | pygame.OPENGL)

        # Initialize OpenGL
        glClearColor(0.5, 0.5, 0.5, 1)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.frame_width / self.frame_height), 0.1, 100.0)
        glEnable(GL_DEPTH_TEST)

        # Set up the camera
        camera_target = [self.camera_position[i] + self.towards_direction[i] for i in range(3)]
        gluLookAt(*self.camera_position, *camera_target, 0, 1, 0)

        # Set up the viewport
        glViewport(0, 0, self.frame_width, self.frame_height)

        # Draw the grid
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(0, 0, 0)
        self.draw_grid(100, 0.5)

        # Capture the frame from OpenGL
        raw_data = glReadPixels(0, 0, self.frame_width, self.frame_height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (self.frame_width, self.frame_height), raw_data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        np_image = np.array(image)

        # Distort the image
        distorted_image = cv2.undistort(np_image, self.camera_matrix, self.dist_coeffs)
        self.image = distorted_image
        pygame.quit()
        return distorted_image
    
    def combine(self, image, alpha1 = 0.5, alpha2 = 0.5, gamma = 0):
        # gamma value (scalar added to each sum)
        return cv2.addWeighted(self.image, alpha1, image, alpha2, gamma)
