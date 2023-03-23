import numpy as np
from typing import Optional


class Vector3:
    def __init__(self, x: float, y: float, z: float):
        """
        Initialize the Vector3 class.
        """
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
    def __init__(self, origin: Vector3, direction: Vector3):
        """
        Initialize the Ray class.
        """
        self.origin = origin
        self.direction = direction.normalized()

class Plane:
    def __init__(self, center: Vector3, normal: Vector3):
        """
        Initialize the Plane class.
        """
        self.center = center
        self.normal = normal.normalized()

    def intersect(self, ray: Ray) -> Optional[Vector3]:
        """
        Calculate the intersection point of the ray and the plane, if it exists.
        """
        if self.normal.dot(ray.direction) != 0:
            t = (self.center - ray.origin).dot(self.normal) / ray.direction.dot(self.normal)
            if t >= 0:
                intersection = ray.origin + ray.direction * t
                return intersection
        return None
