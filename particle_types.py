"""
Generating modelled liquid crystals in 2D
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import numpy as np
from utils import pythagorean_sides


class PointParticle:
    def __init__(self, position: tuple[int, int]):
        """
        A point particle in real space
        :param position: Position of the particle in Cartesian coordinates
        """
        self._position = position  # Position of the particle in real space using cartesian coordinates

    @property
    def position(self):
        return self._position

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The point to be drawn on the real space object
        """
        return draw_object.point(self.position, fill=1)


class CalamiticParticle(PointParticle):
    def __init__(self, init_position: tuple[int, int], width: int, length: int, mean_angle: float, angle_stddev: float):
        """
        A calamitic (rod-like) particle in real space
        :param init_position: Position of the particle in Cartesian coordinates
        :param width: Width of the particle
        :param length: Length of the particle
        :param mean_angle: Angle of the particle in real space
        """
        super().__init__(init_position)
        self._width = width
        self._length = length
        self._angle = np.random.normal(mean_angle, angle_stddev) % 360
        self._get_end_points()

    @property
    def width(self):
        return self._width

    @property
    def length(self):
        return self._length

    @property
    def angle(self):
        return self._angle

    @property
    def end_position(self):
        return self._end_position

    @property
    def x1(self):
        return self.position[0]

    @property
    def y1(self):
        return self.position[1]

    @property
    def x2(self):
        return self._end_position[0]

    @property
    def y2(self):
        return self._end_position[1]

    def _get_end_points(self):
        """
        Calculate the coordinates of the end of the particle, given its length and angle
        :return: The end coordinates of the particle
        """
        x2, y2 = pythagorean_sides(self.length, self.width, self.angle)
        self._end_position = (int(x2) + self.x1, int(y2) + self.y1)

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The line to be drawn on the real space object
        """
        return draw_object.line([self.position, self.end_position], fill=1, width=self.width)
