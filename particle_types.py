"""
Generating modelled liquid crystals in 2D
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import numpy as np


class PointParticle:
    def __init__(self, position: tuple[int, int]):
        """
        A point particle in real space
        :param position: Position of the particle in Cartesian coordinates
        """
        self.position = position  # Position of the particle in real space using cartesian coordinates

    def __get_position__(self):
        return self.position

    def __set_position__(self, new_position: list):
        self.position = new_position

    def __get_x__(self):
        return self.position[0]

    def __set_x__(self, new_x: int):
        self.__set_position__([new_x, self.position[1]])

    def __get_y__(self):
        return self.position[1]

    def __set_y__(self, new_y: int):
        self.__set_position__([self.position[0], new_y])

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The point to be drawn on the real space object
        """
        return draw_object.point(self.position, fill=1)


class CalamiticParticle(PointParticle):
    def __init__(self, init_position: tuple[int, int], width: int, length: int, angle: float):
        """
        A calamitic (rod-like) particle in real space
        :param init_position: Position of the particle in Cartesian coordinates
        :param width: Width of the particle
        :param length: Length of the particle
        :param angle: Angle of the particle in real space
        """
        super().__init__(init_position)
        self.width = width
        self.length = length
        self.size = (self.width, self.length)  # Width and length of the particle
        self.angle = angle
        self.end_position = self.get_end_points()

    def get_width(self):
        return self.size[0]

    def get_len(self):
        return self.size[1]

    def get_angle(self):
        return self.angle

    def get_end_points(self):
        """
        Calculate the coordinates of the end of the particle, given its length and angle
        :return: The end coordinates of the particle
        """
        angle_rad = np.deg2rad(self.angle)
        x1, y1 = self.position
        x2, y2 = [int(x1 + self.get_len() * np.cos(angle_rad)),
                  int(y1 + self.get_len() * np.sin(angle_rad))]
        return x2, y2

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The line to be drawn on the real space object
        """
        return draw_object.line([self.position, self.end_position], fill=1, width=self.get_width())
