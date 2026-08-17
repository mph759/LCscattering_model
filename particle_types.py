"""
Generating modelled liquid crystals in 2D
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
from functools import wraps
from typing import Generator, Any, TypeAlias, Callable, Optional

import numpy as np

Coordinates2D: TypeAlias = tuple[int, int]


class PointParticle:
    def __init__(self, position: tuple[int, int]):
        """
        A point particle in real space
        :param position: Position of the particle in Cartesian coordinates
        """
        self._position = position  # Position of the particle in real space using cartesian coordinates

    @property
    def position(self) -> Coordinates2D:
        return self._position

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def params(self):
        return {'Point Particle', {}}

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The point to be drawn on the real space object
        """
        return draw_object.point(self.position, fill=1)


class CalamiticParticle(PointParticle):
    def __init__(self, init_position: tuple[int, int], width: int, length: int, angle_func: Callable):
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
        self._angle_func = angle_func
        self._set_angle()
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
    def end_position(self) -> Coordinates2D:
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

    @property
    def params(self):
        return {'Calamitic Particle':
                    {'width': self.width,
                     'length': self.length}}

    def _get_end_points(self):
        """
        Calculate the coordinates of the end of the particle, given its length and angle
        :return: The end coordinates of the particle
        """
        x2 = self.x1 + round(self.length * np.cos(np.radians(self.angle)))
        y2 = self.y1 + round(self.length * np.sin(np.radians(self.angle)))
        self._end_position = x2, y2
        self._fix_angel()

    def _set_angle(self):
        angle = self._angle_func()
        while angle < 0:
            angle += 360
        angle %= 360
        self._angle = angle

    def _fix_angel(self):
        if self.x1 == self.x2:
            if self.y1 < self.y2:
                self._angle = 90
            else:
                self._angle = 270
        else:
            self._angle = np.arctan((self.y2 - self.y1) / (self.x2 - self.x1))

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The line to be drawn on the real space object
        """
        return draw_object.line([self.position, self.end_position], fill=1, width=self.width)


def init_spacing(particle_length: int, particle_width: int,
                 unit_vector: int, padding_spacing: Coordinates2D) -> tuple[Coordinates2D, Coordinates2D]:
    """
    Initializes the spacing of the particles based on the particle length and particle width
    :param particle_length:
    :param particle_width:
    :param unit_vector:
    :param padding_spacing:
    :return:
    """
    x_spacing, y_spacing = (spacing + padding
                            for spacing, padding
                            in zip(pythagorean_sides(particle_length, particle_width, unit_vector), padding_spacing))

    # Allow for particles to move slightly in x and y, depending on the spacing
    displacement = tuple([np.ceil(spacing / 2) for spacing in padding_spacing])
    print(f'x spacing: {x_spacing}, y spacing: {y_spacing}')
    print(f'displacement: {displacement}')
    spacing = (x_spacing, y_spacing)
    return spacing, displacement


def generate_positions(space: Coordinates2D, maximum: Coordinates2D,
                       change: Coordinates2D) -> Generator[Coordinates2D, Any, None]:
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spatial oscillations
    :rtype: Generator[
    Coordinates2D, Any, None]
    :param space: Spacing in x and y-dimensions between positions
    :param maximum: Maximum values in x and y-dimensions
    :param change: Tuple of allowed deviation from initial lattice spacing
    :return: A position in Cartesian coordinates inside the grid
    """
    # Initial positions, just inside the box
    x_space, y_space = space
    x_change, y_change = np.abs(change)
    x_max, y_max = maximum
    x = int(x_space / 2)
    y = int(y_space / 2)

    # Loop while the positions are still inside the box
    while y < y_max:
        x_pos = x
        y_pos = y
        if x_change != 0:
            x_pos += np.random.randint(-x_change, x_change)
        if y_change != 0:
            y_pos += np.random.randint(-y_change, y_change)
        yield x_pos, y_pos
        x += x_space

        # When the position is at the edge of the box, adjust y and reset x
        if x >= x_max:
            y += y_space
            x = x_space

def generate_random_positions(num_particles: int, maximum: Coordinates2D) -> Generator[Coordinates2D, Any, None]:
    for _ in range(num_particles):
        yield np.random.randint(0, maximum[0]), np.random.randint(0, maximum[1])


def normal_distribution(mean, stddev):
    return np.random.normal(mean, stddev)

def uniform_distribution(min, max):
    return np.random.uniform(min, max)


def exact(value):
    return value


def pythagorean_sides(a: float | int, b: float | int, theta: float | int) -> tuple[int, int]:
    """
    Calculates the side lengths of a right angle triangle using the Pythagorean formulae
    :param a: Length of triangle (a)
    :param b: Width of the triangle (b)
    :param theta: Angle of the triangle
    :return: x and y coordinates of the end point
    """
    theta_radians = np.deg2rad(theta)
    x = abs(np.round(a * np.cos(theta_radians))) + abs(np.round(b * np.sin(theta_radians)))
    y = abs(np.round(a * np.sin(theta_radians))) + abs(np.round(b * np.cos(theta_radians)))
    return x, y


