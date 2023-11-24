"""
Generating 2D scattering pattern for modelled liquid crystals
Original Author: Campbell Tims
Edited by Michael Hassett from 2023-11-23
"""

import numpy as np
import matplotlib.pyplot as plt
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


class PointParticle:
    def __init__(self, position: list[int, int]):
        '''
        A point particle in real space
        :param position: Position of the particle in Cartesian coordinates
        '''
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


class CalamiticParticle(PointParticle):
    def __init__(self, position, width, length, angle):
        """
        A calamitic (rod-like) particle in real space
        :param position: Position of the particle in Cartesian coordinates
        :param width: Width of the particle
        :param length: Length of the particle
        :param angle: Angle of the particle in real space
        """
        super().__init__(position)
        self.size = (width, length)  # Width and length of the particle
        self.__set_angle__(angle)

    def __get_wid__(self):
        return self.size[0]

    def __get_len__(self):
        return self.size[1]

    def __len__(self):
        return self.__get_len__()

    def __get_angle__(self):
        return self.angle

    def __set_angle__(self, input_angle):
        self.angle = input_angle


def generate_positions(x_change, y_change):
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spacial oscillations
    :param x_change: Maximum change in x
    :param y_change: Maximum change in y
    :return: A position in Cartesian coordinates inside the grid
    """
    # Initial positions, just inside the box
    x = x_spacing
    y = y_spacing

    # Loop while the positions are still inside the box
    while y < y_max:
        x_pos = x
        y_pos = y
        if x_change != 0:
            x_pos += np.random.randint(-x_change, x_change)
        if y_change != 0:
            y_pos += np.random.randint(-y_change, y_change)
        yield [x_pos, y_pos]
        x += x_spacing

        # When the position is at the edge of the box, adjust y and reset x
        if x >= x_max:
            y += y_spacing
            x = x_spacing


def generate_angles(mean_angle: int, angle_stddev: int):
    """
    Generate an angle from a normal distribution, with a given mean and standard deviation
    :param mean_angle: Mean value of the angle
    :param angle_stddev: standard deviation for the angle
    :return:
    """
    angle = mean_angle
    while True:
        angle = np.random.normal(mean_angle, angle_stddev)
        if angle >= 360:
            angle -= 360
        if angle < 0:
            angle += 360
        yield angle


if __name__ == "__main__":
    # Initialise real space grid
    x_max = y_max = 1000
    grid_size = (x_max, y_max)
    real_space = np.zeros(grid_size)

    # Initialise single particle parameters
    particle_width = 2
    particle_length = 15
    unit_vector = 90  # unit vector of the particle, starting point up
    vector_range = 45  # Full angular range for the unit vector
    vector_min, vector_max = (unit_vector + change for change in (-vector_range / 2, vector_range / 2))
    unit_vector = np.random.randint(vector_min, vector_max)
    print(f'Unit Vector: {unit_vector}\N{DEGREE SIGN}')
    # Note: The unit vector is not the exact angle all the particles will have, but the mean of all the angles

    # Initialise how the particles sit in real space
    standard_spacing = 3
    x_spacing = particle_width + standard_spacing
    y_spacing = int(np.ceil(particle_length * np.cos(np.deg2rad(unit_vector - 90)))) + standard_spacing
    print(f'x spacing: {x_spacing}, y spacing: {y_spacing}')

    # Generate the particles
    # particles = CalamiticParticle(0, 0, particle_width, particle_length, unit_vector)
    positions = generate_positions(2, 2)
    angles = generate_angles(unit_vector, 5)
    particles = [CalamiticParticle(position, particle_width, particle_length, angle)
                 for position, angle in zip(positions, angles)]
    print(f'No. of Particles: {len(particles)}')
