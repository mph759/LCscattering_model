"""
Generating 2D scattering pattern for modelled liquid crystals
Original Author: Campbell Tims
Edited by Michael Hassett from 2023-11-23
"""

import numpy as np
from PIL import Image
from PIL import ImageDraw
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
    def __init__(self, init_position, width, length, angle):
        """
        A calamitic (rod-like) particle in real space
        :param position: Position of the particle in Cartesian coordinates
        :param width: Width of the particle
        :param length: Length of the particle
        :param angle: Angle of the particle in real space
        """
        super().__init__(init_position)
        self.size = (width, length)  # Width and length of the particle
        self.__set_angle__(angle)
        self.end_position = self.get_end_points()

    def get_width(self):
        return self.size[0]

    def __get_len__(self):
        return self.size[1]

    def __len__(self):
        return self.__get_len__()

    def __get_angle__(self):
        return self.angle

    def __set_angle__(self, input_angle):
        self.angle = input_angle

    def get_end_points(self):
        angle_rad = np.deg2rad(self.angle)
        x1, y1 = self.position
        x2, y2 = [int(x1 + self.__get_len__() * np.cos(angle_rad)),
                  int(y1 + self.__get_len__() * np.sin(angle_rad))]
        return x2, y2


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
        yield x_pos, y_pos
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
    while True:
        angle = angle_correction(np.random.normal(mean_angle, angle_stddev))
        yield angle


def angle_correction(angle):
    while angle > 360:
        angle -= 360
    while angle < 0:
        angle += 360
    return angle


if __name__ == "__main__":
    # Initialise real space grid
    x_max = y_max = 1000
    grid_size = (x_max, y_max)
    real_space = Image.new('L', grid_size, 0)
    add_to_real_space = ImageDraw.Draw(real_space)

    # Initialise single particle parameters
    particle_width = 2
    particle_length = 15
    unit_vector = 90  # unit vector of the particle, starting point up
    vector_range = 40  # Full angular range for the unit vector
    vector_min, vector_max = (unit_vector + change for change in (-vector_range / 2, vector_range / 2))
    unit_vector = np.random.randint(vector_min, vector_max)
    # print(f'Min. Angle: {vector_min}\N{DEGREE SIGN}, Max. Angle: {vector_max}\N{DEGREE SIGN}')
    print(f'Unit Vector: {unit_vector}\N{DEGREE SIGN}')
    # Note: The unit vector is not the exact angle all the particles will have, but the mean of all the angles

    # Initialise how the particles sit in real space
    unit_vector_radians = np.deg2rad(unit_vector)
    standard_spacing = 5
    # Allow spacing in x and y to account for the size and angle of the particle
    x_spacing = standard_spacing + (int(abs(np.round(particle_length * np.cos(unit_vector_radians)))) +
                                    int(abs(np.round(particle_width * np.sin(unit_vector_radians)))))
    y_spacing = standard_spacing + (int(abs(np.round(particle_length * np.sin(unit_vector_radians)))) +
                                    int(abs(np.round(particle_width * np.cos(unit_vector_radians)))))
    print(f'x spacing: {x_spacing}, y spacing: {y_spacing}')

    # Generate the particles
    positions = generate_positions(2, 2)
    angles = generate_angles(unit_vector, 5)
    particles = [CalamiticParticle(position, particle_width, particle_length, angle)
                 for position, angle in zip(positions, angles)]
    print(f'No. of Particles: {len(particles)}')
    for particle in particles:
        #print(f'Start: {particle.position}, End: {particle.end_position}')
        add_to_real_space.line([particle.position, particle.end_position], fill=1, width=particle.get_width())
    real_space = np.asarray(real_space)

    # Plot real_space
    plt.figure(figsize=(8, 8))
    plt.imshow(real_space, extent=[0, x_max, 0, y_max])
    plt.title(f'Liquid Crystal Phase of Calamitic Liquid crystals, with unit vector {unit_vector}$^\circ$')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
