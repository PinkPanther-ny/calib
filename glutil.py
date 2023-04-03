from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
import numpy as np


class GLGrid:
    def __init__(self, size, spacing, colored=True):
        self.size = size
        self.spacing = spacing
        self.colored = colored

    def draw(self, colored=True):
        if colored:
            # Define the colors for the gradient
            colors = [
                (1, 0, 0),  # Red
                (0, 1, 0),  # Green
                (0, 0, 1),  # Blue
            ]

            max_distance = self.size * self.spacing

            # Draw filled quads for each grid cell
            glBegin(GL_QUADS)
            for i in range(-self.size, self.size):
                for j in range(-self.size, self.size):
                    for x, z in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]:
                        # Calculate the distance from the center
                        distance = np.sqrt(x * x + z * z)

                        # Calculate the color based on the distance
                        t = distance / max_distance
                        t = min(1, max(0, t))

                        if t < 0.5:
                            t = t * 2
                            color = (
                                colors[0][0] * (1 - t) + colors[1][0] * t,
                                colors[0][1] * (1 - t) + colors[1][1] * t,
                                colors[0][2] * (1 - t) + colors[1][2] * t,
                            )
                        else:
                            t = (t - 0.5) * 2
                            color = (
                                colors[1][0] * (1 - t) + colors[2][0] * t,
                                colors[1][1] * (1 - t) + colors[2][1] * t,
                                colors[1][2] * (1 - t) + colors[2][2] * t,
                            )

                        glColor3f(*color)
                        glVertex3f(x * self.spacing, 0, z * self.spacing)
            glEnd()

        # Draw the grid lines
        glBegin(GL_LINES)
        glColor3f(0, 0, 0)
        for i in range(-self.size, self.size + 1):
            glVertex3f(i * self.spacing, 0.001, -self.size * self.spacing)
            glVertex3f(i * self.spacing, 0.001, self.size * self.spacing)
            glVertex3f(-self.size * self.spacing, 0.001, i * self.spacing)
            glVertex3f(self.size * self.spacing, 0.001, i * self.spacing)
        glEnd()

class GLCompass:
    def __init__(self, frame_width, frame_height, x_spacing=10, y_loc=20, colored=True):
        self.compass_labels = {
            'N': 0,
            'NE': 45,
            'E': 90,
            'SE': 135,
            'S': 180,
            'SW': 225,
            'W': 270,
            'NW': 315
        }
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.x_spacing = x_spacing
        self.y_loc = y_loc

    def draw(self, towards_direction):
        x, _, z = towards_direction
        angle_rad = np.arctan2(-x, z)  # Invert the x-axis angle calculation
        angle_deg = np.degrees(angle_rad)
        compass_angle = (angle_deg + 360) % 360
        
        
        screen_center_x = self.frame_width // 2

        for label, direction_angle in self.compass_labels.items():
            # Calculate the difference between the direction_angle and compass_angle, considering the edges
            diff_angle = (direction_angle - compass_angle + 180) % 360 - 180

            # Calculate the x position based on the diff_angle
            offset_x = diff_angle * self.x_spacing
            x = screen_center_x + offset_x
            y = self.frame_height - self.y_loc

            draw_text(label, x, y, self.frame_width, self.frame_height, align_center=True)

        angle_text = f"{compass_angle:.1f}°"
        draw_text(angle_text, screen_center_x, self.frame_height - self.y_loc - 15, self.frame_width, self.frame_height, align_center=True)



def create_texture_from_text(text, font_size, color):
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

def draw_text(text, x, y, frame_width, frame_height, align_center=False, font_size=18, font_color=(255, 255, 255)):
    texture_id, width, height = create_texture_from_text(text, font_size, font_color)
    if align_center:
        x = x - width // 2
    glColor3f(0,0,0)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, frame_width, 0, frame_height)
    
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