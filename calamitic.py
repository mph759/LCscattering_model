"""
Generating 2D scattering pattern for modelled liquid crystals
Original Author: Campbell Tims
Edited by Michael Hassett from 2023-11-23
"""

import time
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


class RealSpace:
    def __init__(self, grid):
        self.grid = grid
        self.img = Image.new('L', self.grid, 0)
        self.array = np.asarray(self.img)

    def add(self, particle_list):
        in_real_space = ImageDraw.Draw(self.img)
        for particle in particle_list:
            # print(f'Start: {particle.position}, End: {particle.end_position}')
            particle.create(in_real_space)
        self.__set_array__()

    def __set_array__(self):
        self.array = np.asarray(self.img)

    def plot(self, title, **kwargs):
        """
        Plot the 2D real space image
        :param title: Title text for figure
        :return: Figure object
        """
        plt.figure()
        plt.imshow(self.array, extent=(0, self.grid[0], 0, self.grid[1]))
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        if 'show' in kwargs and kwargs['show']:
            plt.show()


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

    def create(self, draw_object):
        return draw_object.point(self.position, fill=1)


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

    def create(self, draw_object):
        return draw_object.line([self.position, self.end_position], fill=1, width=self.get_width())


def generate_positions(x_change, y_change):
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spacial oscillations
    :param x_change: Maximum change in x
    :param y_change: Maximum change in y
    :return: A position in Cartesian coordinates inside the grid
    """
    # Initial positions, just inside the box
    x = int(x_spacing / 2)
    y = int(y_spacing / 2)

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
        angle = np.random.normal(mean_angle, angle_stddev) % 360
        yield angle


def pythagorean_sides(a, b, theta):
    theta_radians = np.deg2rad(theta)
    x = (abs(np.round(a * np.cos(theta_radians))) + abs(np.round(b * np.sin(theta_radians))))
    y = (abs(np.round(a * np.sin(theta_radians))) + abs(np.round(b * np.cos(theta_radians))))
    return x, y


class DiffractionPattern:
    def __init__(self, space_object: RealSpace, wavelength, pixel_size, npt):
        self.space = space_object
        self.pattern_2d = self.create_2d_diffraction(self.space)
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pattern_1d = self.create_1d_diffraction(npt)

    def create_2d_diffraction(self, space_object):
        '''
        Create the diffraction pattern from the given real space object
        :param space_object: RealSpace object
        :return: Diffraction pattern of the real space object
        '''
        fourier_of_space = np.fft.fft2(space_object.array)
        fourier_of_space = np.roll(fourier_of_space, space_object.grid[0] // 2, 0)
        fourier_of_space = np.roll(fourier_of_space, space_object.grid[1] // 2, 1)

        diffraction_image = np.abs(fourier_of_space)

        # Elimintate the centre pixel
        diffraction_image[space_object.grid[1] // 2][space_object.grid[0] // 2] = 0
        return diffraction_image

    def plot_2d(self, title, **kwargs):
        # Plot the diffraction image
        plt.figure()
        plt.imshow(self.pattern_2d ** 2)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        if 'clim' in kwargs:
            plt.clim(0, kwargs['clim'])
        if 'show' in kwargs and kwargs['show']:
            plt.show()

    def frm_integration(self, frame, unit="q_nm^-1", npt=2250):
        """
        Perform azimuthal integration of frame array
        :param frame: numpy array containing 2D intensity
        :param unit:
        :param npt:
        :return: two-col array of q & intensity.
        """
        # print("Debug - ", self.cam_length, self.pixel_size, self.wavelength)
        cam_length = self.pixel_size * self.space.grid[0] / self.wavelength
        image_center = (self.space.grid[0] / 2, self.space.grid[1] / 2)
        ai = AzimuthalIntegrator()
        ai.setFit2D(directDist=cam_length / 1000,
                    centerX=image_center[0],
                    centerY=image_center[1],
                    pixelX=self.pixel_size, pixelY=self.pixel_size)
        ai.wavelength = self.wavelength
        integrated_profile = ai.integrate1d(data=frame, npt=npt, unit=unit)
        return np.transpose(np.array(integrated_profile))

    def create_1d_diffraction(self, npt):
        radius = self.space.grid[0] // 2  # Assuming you have defined xmax and ymax somewhere
        diffraction_image_cone = circular_mask(self.space.grid, radius) * self.pattern_2d
        diffraction_plot = self.frm_integration(diffraction_image_cone, unit="q_nm^-1", npt=npt)

        non_zero = diffraction_plot[:, 1] != 0  # Removes data points which = 0 due to the cone restriction
        diffraction_plot_filtered = diffraction_plot[non_zero]
        return diffraction_plot_filtered

    def plot_1d(self, title, **kwargs):
        """
        Plot a 1D diffraction pattern
        :param title: Title text for the plotting
        :param kwargs:
        :return:
        """

        # Plot 1D integration
        plt.figure()
        plt.title(title)
        plt.plot(self.pattern_1d[int(npt // 20):, 0], self.pattern_1d[int(npt // 20):, 1])
        plt.xlabel(f'q / nm$^{-1}$')
        plt.ylabel('Arbitrary Intensity')
        plt.tight_layout()
        if 'show' in kwargs and kwargs['show']:
            plt.show()


def circular_mask(grid, mask_radius):
    kernel = np.zeros(grid)
    filter_y, filter_x = np.ogrid[-mask_radius:mask_radius, -mask_radius:mask_radius]
    mask = filter_x ** 2 + filter_y ** 2 <= mask_radius ** 2
    kernel[mask] = 1
    return kernel


if __name__ == "__main__":
    tic = time.perf_counter()  # Start timer

    # Initialise real space parameters
    x_max = y_max = 1000
    grid_size = (x_max, y_max)

    # Initialise particle parameters
    particle_width = 2
    particle_length = 15
    unit_vector = 90  # unit vector of the particle, starting point up
    vector_range = 40  # Full angular range for the unit vector
    vector_stddev = 5

    # Initialise how the particles sit in real space
    standard_spacing = 6
    wobble_allowance = np.floor((standard_spacing - 1) / 2)

    # Randomise unit_vector within given range
    vector_min, vector_max = (unit_vector + change for change in (-vector_range / 2, vector_range / 2))
    #unit_vector = np.random.randint(vector_min, vector_max) % 360
    # print(f'Min. Angle: {vector_min}\N{DEGREE SIGN}, Max. Angle: {vector_max}\N{DEGREE SIGN}')
    print(f'Unit Vector: {unit_vector}\N{DEGREE SIGN}')
    # Note: The unit vector is not the exact angle all the particles will have, but the mean of all the angles

    # Allow spacing in x and y to account for the size and angle of the particle
    x_spacing, y_spacing = pythagorean_sides(particle_length, particle_width, unit_vector)
    x_spacing += standard_spacing
    y_spacing += standard_spacing
    print(f'x spacing: {x_spacing}, y spacing: {y_spacing}')

    # Generate the particles
    positions = generate_positions(wobble_allowance, wobble_allowance)
    angles = generate_angles(unit_vector, vector_stddev)
    particles = [CalamiticParticle(position, particle_width, particle_length, angle)
                 for position, angle in zip(positions, angles)]
    print(f'No. of Particles: {len(particles)}')

    # Place particles in real space
    real_space = RealSpace(grid_size)
    real_space.add(particles)
    toc = time.perf_counter()
    print(f'Generating the particles in real space took {toc - tic:0.4f} seconds')
    real_space_title = f'Liquid Crystal Phase of Calamitic Liquid crystals, with unit vector {unit_vector}$^\circ$'
    real_space.plot(real_space_title)

    tic = time.perf_counter()
    wavelength = 1e-10
    pixel_size = 5e-5
    npt = 2000
    diffraction_pattern_of_real_space = DiffractionPattern(real_space, wavelength, pixel_size, npt)
    diffraction_pattern_title = f'Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
    diffraction_pattern_of_real_space.plot_2d(diffraction_pattern_title, clim=1e7)
    diff_1D_title = f'1D Radial Integration'
    diffraction_pattern_of_real_space.plot_1d(diff_1D_title)
    toc = time.perf_counter()
    print(f'Generating the diffraction pattern took {toc - tic:0.4f} seconds')
    plt.show()
