import numpy as np
from typing import Optional


class Vector3(np.ndarray):
    def __new__(cls, x: float, y: float, z: float):
        return np.asarray([x, y, z]).view(cls)

    def __init__(self, x: float, y: float, z: float):
        pass

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @property
    def length(self):
        return np.linalg.norm(self.array)

    def normalized(self):
        return self / np.linalg.norm(self)

    def __str__(self) -> str:
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
        t = np.dot(self.center - ray.origin, self.normal) / np.dot(ray.direction, self.normal)

        intersections = ray.origin + t[..., np.newaxis] * ray.direction
        intersections[t <= 0] = [float('inf'), float('inf'), float('inf')]

        return intersections

    def intersect_many(self, ray_origin: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
        """
        Calculate the intersection points of a set of rays to the plane
        """
        t = np.dot(self.center - ray_origin, self.normal) / np.dot(ray_directions, self.normal)

        intersections = ray_origin + t[..., np.newaxis] * ray_directions
        intersections[t <= 0] = [float('inf'), float('inf'), float('inf')]

        return intersections

