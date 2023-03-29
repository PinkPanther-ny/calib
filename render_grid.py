import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import pygame
from core import Calib

def draw_grid(size, spacing):
    glBegin(GL_LINES)
    for i in range(-size, size + 1):
        glVertex3f(i * spacing, 0, -size * spacing)
        glVertex3f(i * spacing, 0, size * spacing)
        glVertex3f(-size * spacing, 0, i * spacing)
        glVertex3f(size * spacing, 0, i * spacing)
    glEnd()

def render_image(camera_matrix, dist_coeffs, width, height, camera_position, towards_direction):
    # Initialize Pygame
    pygame.init()
    pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)

    # Initialize OpenGL
    glClearColor(0.5, 0.5, 0.5, 1)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (width / height), 0.1, 100.0)
    glEnable(GL_DEPTH_TEST)

    # Set up the camera
    camera_target = [camera_position[i] + towards_direction[i] for i in range(3)]
    gluLookAt(*camera_position, *camera_target, 0, 1, 0)

    # Set up the viewport
    glViewport(0, 0, width, height)

    # Draw the grid
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor3f(0, 0, 0)
    draw_grid(20, 0.5)

    # Capture the frame from OpenGL
    raw_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), raw_data).transpose(Image.FLIP_TOP_BOTTOM)
    np_image = np.array(image)

    # Distort the image
    distorted_image = cv2.undistort(np_image, camera_matrix, dist_coeffs)

    return distorted_image


if __name__ == "__main__":
    calib = Calib(filename="configs/calib_data.json")
    rendered_image = render_image(calib.camera_matrix, calib.dist_coeffs, 
                 calib.width, calib.height, 
                 [0, 0.5, 0.0], [0, 0, 1])

    # Save the rendered image
    cv2.imwrite("rendered_image.png", cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
