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
        self.towards_direction = np.array(towards_direction)
        
        # Initialize Pygame, pygame.OPENGL indicates that the window should be created with an OpenGL context.
        pygame.init()
        pygame.display.set_mode((self.frame_width, self.frame_height), pygame.DOUBLEBUF | pygame.OPENGL)
        self.compute()

    def draw_compass(self, x, y, width, height):
        # Calculate the angle between the camera's forward direction and the world's north direction (Z axis)
        north_direction = np.array([0, 0, 1])
        current_direction = self.towards_direction.copy()
        current_direction[1] = 0
        current_direction = current_direction / np.linalg.norm(current_direction)
        angle = np.degrees(np.arccos(np.clip(np.dot(north_direction, current_direction), -1.0, 1.0)))

        if self.towards_direction[0] < 0:
            angle = 360 - angle

        # Draw the compass background
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()

        # Draw compass ticks and labels
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        tick_interval = 45
        tick_length = 10

        for i, label in enumerate(labels):
            tick_angle = i * tick_interval - angle
            tick_x = x + width / 2 + (width / 2 - tick_length / 2) * np.cos(np.radians(tick_angle))
            tick_y = y + height / 2 + (height / 2 - tick_length / 2) * np.sin(np.radians(tick_angle))

            glColor3f(1, 1, 1)
            glBegin(GL_LINES)
            glVertex2f(tick_x, tick_y)
            glVertex2f(tick_x + tick_length * np.cos(np.radians(tick_angle)), tick_y + tick_length * np.sin(np.radians(tick_angle)))
            glEnd()

            label_texture_id, label_width, label_height = self.create_texture_from_text(label, 18, (255, 255, 255))
            label_x = tick_x + (tick_length + 5) * np.cos(np.radians(tick_angle)) - label_width / 2
            label_y = tick_y + (tick_length + 5) * np.sin(np.radians(tick_angle)) - label_height / 2
            self.draw_text(label_texture_id, label_x, label_y, label_width, label_height)

        # Draw the compass needle
        glColor3f(1, 0, 0)
        glBegin(GL_TRIANGLES)
        glVertex2f(x + width / 2, y + height / 2)
        glVertex2f(x + width / 2 - 5, y + height)
        glVertex2f(x + width / 2 + 5, y + height)
        glEnd()

    def draw_fade_circle(self, center_color, outside_color, radius, num_segments):
        glBegin(GL_TRIANGLE_FAN)
        
        glColor3f(*center_color)
        glVertex3f(0, -0.05, 0)
        
        angle_step = 2 * np.pi / num_segments
        for i in range(num_segments + 1):
            angle = i * angle_step
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)

            glColor3f(*outside_color)
            glVertex3f(x, 0.001, z)

        glEnd()

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

        move_speed = 0.02
        mouse_sensitivity = 0.2

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            
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
                move_vector_normalized = [coord / move_vector_length for coord in move_vector]
            else:
                move_vector_normalized = move_vector

            # Update camera position with normalized movement vector
            self.camera_position[0] += move_speed * move_vector_normalized[0]
            self.camera_position[2] += move_speed * move_vector_normalized[2]

            if keys[pygame.K_ESCAPE]:
                running = False


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

            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set up the camera
            camera_target = [self.camera_position[i] + self.towards_direction[i] for i in range(3)]
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(*self.camera_position, *camera_target, 0, 1, 0)

            glColor3f(0, 0, 0)
            self.draw_grid(100, 0.5)
            self.draw_fade_circle(center_color=(1, 0, 0), outside_color=(0, 1, 0), radius=10, num_segments=100)

            # Draw text on the screen
            camera_position_text = f"Camera Position: {np.round(self.camera_position, 2)}"
            towards_direction_text = f"Towards Direction: {np.round(self.towards_direction, 2)}"
            
            camera_position_texture_id, cam_pos_width, cam_pos_height = self.create_texture_from_text(camera_position_text, 18, (255, 255, 255))
            towards_direction_texture_id, towards_dir_width, towards_dir_height = self.create_texture_from_text(towards_direction_text, 18, (255, 255, 255))
            
            self.draw_text(camera_position_texture_id, 10, 30, cam_pos_width, cam_pos_height)
            self.draw_text(towards_direction_texture_id, 10, 10, towards_dir_width, towards_dir_height)
            
            self.draw_compass(self.frame_width // 2 - 100, self.frame_height - 40, 200, 20)

            pygame.display.flip()
            clock.tick(60)
        pygame.quit()
        
    def create_texture_from_text(self, text, font_size, color):
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, color)
        text_surface = pygame.transform.flip(text_surface, False, True)
        width, height = text_surface.get_size()
        text_data = np.frombuffer(text_surface.get_buffer(), np.uint8)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        return texture_id, width, height

    def draw_text(self, texture_id, x, y, width, height):
        glColor3f(0,0,0)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.frame_width, 0, self.frame_height)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + width, y)
        glTexCoord2f(1, 1)
        glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1)
        glVertex2f(x, y + height)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()


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