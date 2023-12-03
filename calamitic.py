"""
Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""

import time
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import inspect

def timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        result = func(*args, **kwargs)
        t_end = time.perf_counter()
        t_total = t_end - t_start
        func_description = inspect.getdoc(func).split('\n')[0]
        print(f'{func_description} took {t_total:0.4f}s')
        return result
    return wrapper


class RealSpace:
    def __init__(self, grid):
        self.grid = grid
        print(f"Generating real space array with dimensions {self.grid}")
        self.img = Image.new('L', self.grid, 0)
        self.__set_array__()

    @timer
    def add(self, particle_list):
        """Adding particles to real space"""
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
        :return:
        """
        print("Plotting real space figure...")
        plt.figure(figsize=figure_size)
        plt.imshow(self.array, extent=(0, self.grid[0], 0, self.grid[1]))
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        if 'show' in kwargs and kwargs['show']:
            plt.show()


class PointParticle:
    def __init__(self, position: list[int, int]):
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
    def __init__(self, init_position, width, length, angle):
        """
        A calamitic (rod-like) particle in real space
        :param init_position: Position of the particle in Cartesian coordinates
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

    def get_len(self):
        return self.size[1]

    def get_angle(self):
        return self.angle

    def __set_angle__(self, input_angle):
        self.angle = input_angle

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


def generate_positions(change):
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spacial oscillations
    :param change: Tuple of allowed change in x and y
    :return: A position in Cartesian coordinates inside the grid
    """
    # Initial positions, just inside the box
    x_change = change[0]
    y_change = change[1]
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
    :param init_angle: Mean value of the angle
    :param angle_range: Range for starting angle
    :param angle_stddev: standard deviation for the angle
    :return:
    """
    while True:
        angle = np.random.normal(mean_angle, angle_stddev) % 360
        yield angle


def initialise_angle(init_angle, angle_range):
    if angle_range != 0:
        angle_min, angle_max = (init_angle + change for change in (-angle_range / 2, angle_range / 2))
        mean_angle = np.random.randint(angle_min, angle_max) % 360  # Randomise unit vector in given range
        # print(f'Min. Angle: {vector_min}\N{DEGREE SIGN}, Max. Angle: {vector_max}\N{DEGREE SIGN}')
    else:
        mean_angle = init_angle
    print(f'Unit Vector: {mean_angle}\N{DEGREE SIGN}')
    return mean_angle


def pythagorean_sides(a, b, theta):
    """
    Calculates the side lengths of a right angle triangle using the Pythagorean formulae
    :param a: Length of triangle (a)
    :param b: Width of the triangle (b)
    :param theta: Angle of the triangle
    :return: x and y coordinates of the end point
    """
    theta_radians = np.deg2rad(theta)
    x = (abs(np.round(a * np.cos(theta_radians))) + abs(np.round(b * np.sin(theta_radians))))
    y = (abs(np.round(a * np.sin(theta_radians))) + abs(np.round(b * np.cos(theta_radians))))
    return x, y


class DiffractionPattern:
    @timer
    def __init__(self, space_object: RealSpace, wavelength, pixel_size, npt):
        """
        Generating 1D and 2D diffraction patterns on a real space object
        :param space_object: RealSpace object with particles to be diffracted
        :param wavelength: Wavelength of the beam
        :param pixel_size: Size of the pixels on the simulated detector
        :param npt: Number of points in radial dimension
        """
        self.space = space_object
        self.pattern_2d = self.create_2d_diffraction()
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pattern_1d = self.create_1d_diffraction(npt)

    def create_2d_diffraction(self):
        """
        Simulate the 2D diffraction from the given real space object
        :return: Diffraction pattern of the real space object
        """
        print("Generating 2D diffraction image...")
        fourier_of_space = np.fft.fft2(self.space.array)
        fourier_of_space = np.roll(fourier_of_space, self.space.grid[0] // 2, 0)
        fourier_of_space = np.roll(fourier_of_space, self.space.grid[1] // 2, 1)

        diffraction_image = np.abs(fourier_of_space)

        # Eliminate the centre pixel
        diffraction_image[self.space.grid[1] // 2][self.space.grid[0] // 2] = 0
        print("2D diffraction image complete")
        return diffraction_image

    def plot_2d(self, title, **kwargs):
        """
        Plot the 2D Diffraction image
        :param title: String to be placed as a title on the figure
        :param clim (Optional): Colour bar limit
        :param show (Optional): Boolean for whether to show the plot immediately. Default False
        :return:
        """
        print("Plotting 2D diffraction figure...")
        # Plot the diffraction image
        plt.figure(figsize=figure_size)
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
        :param unit: Unit used in the radial integration
        :param npt: Number of points used in the radial integration
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
        """
        Simulate the 1D diffraction from the given real space object, through radial integration
        :param npt: Number of points used in the radial integration
        :return:
        """
        print("Generating 1D diffraction image...")
        radius = self.space.grid[0] // 2
        diffraction_image_cone = circular_mask(self.space.grid, radius) * self.pattern_2d
        diffraction_plot = self.frm_integration(diffraction_image_cone, unit="q_nm^-1", npt=npt)

        non_zero = diffraction_plot[:, 1] != 0  # Removes data points at = 0 due to the cone restriction
        diffraction_plot_filtered = diffraction_plot[non_zero]
        print("1D diffraction image complete")
        return diffraction_plot_filtered

    def plot_1d(self, title, **kwargs):
        """
        Plot a 1D diffraction pattern
        :param title: Title text for the plotting
        :param show (Optional): Boolean for whether to show the plot immediately. Default False
        :return:
        """
        # Plot 1D integration
        print("Plotting 1D diffraction figure...")
        plt.figure(figsize=figure_size)
        plt.title(title)
        plt.plot(self.pattern_1d[int(npt // 20):, 0], self.pattern_1d[int(npt // 20):, 1])
        plt.xlabel(f'q')  # / nm$^{-1}$')
        plt.ylabel('Arbitrary Intensity')
        plt.tight_layout()
        if 'show' in kwargs and kwargs['show']:
            plt.show()

    def save_1d(self, file_name):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :return:
        """
        full_file_name = f'{file_name}.npy'
        np.save(full_file_name, self.pattern_1d)
        print(f'Saved 1D diffraction pattern as {full_file_name}')


def circular_mask(grid, mask_radius, **kwargs):
    """
    Create a circular mask over an image
    :param grid: x and y grid size which the mask will fit over
    :param mask_radius: Radius that the mask should sit on
    :return:
    """
    kernel = np.zeros(grid)
    filter_y, filter_x = np.ogrid[-mask_radius:mask_radius, -mask_radius:mask_radius]
    mask = filter_x ** 2 + filter_y ** 2 <= mask_radius ** 2
    kernel[mask] = 1
    if 'show' in kwargs and kwargs['show']:
        plt.figure(figsize=figure_size)
        plt.imshow(kernel)
        plt.plot()
    return kernel


if __name__ == "__main__":
    # Initialise real space parameters
    x_max = y_max = int(1e4)

    # Initialise particle parameters
    particle_width = 2
    particle_length = 15
    # Note: The unit vector is not the exact angle all the particles will have, but the mean of all the angles
    unit_vector = 90  # Unit vector of the particles, starting point up
    vector_range = 40  # Full angular range for the unit vector to be randomised in
    vector_stddev = 5  # Standard Deviation of the angle, used to generate angles for individual particles

    # Initialise how the particles sit in real space
    padding_spacing = (5, 5)

    # Initialise beam and detector parameters
    wavelength = 0.67018e-10
    pixel_size = 75e-6
    npt = 2000

    figure_size = (10, 10)
    ##################### ONLY MODIFY ABOVE #####################
    # Allow spacing in x and y to account for the size and angle of the particle
    unit_vector = initialise_angle(unit_vector, vector_range)
    angles = generate_angles(unit_vector, vector_stddev)
    x_spacing, y_spacing = (spacing + padding
                            for spacing, padding
                            in zip(pythagorean_sides(particle_length, particle_width, unit_vector), padding_spacing))
    print(f'x spacing: {x_spacing}, y spacing: {y_spacing}')

    # Allow for particles to move slightly in x and y, depending on the spacing
    wobble_allowance = tuple([np.floor((spacing - 1) / 2) for spacing in padding_spacing])
    positions = generate_positions(wobble_allowance)

    # Create the space for the particles
    real_space = RealSpace((x_max, y_max))

    # Generate the particles
    particles = [CalamiticParticle(position, particle_width, particle_length, angle)
                 for position, angle in zip(positions, angles)]
    print(f'No. of Particles: {len(particles)}')
    # Check unit vector matches expected value
    particles_unit_vector = np.mean([particle.angle for particle in particles])
    # print(f"Collective unit vector: {particles_unit_vector:0.2f}")

    # Place particles in real space
    real_space.add(particles)
    real_space_title = f'Liquid Crystal Phase of Calamitic Liquid crystals, with unit vector {unit_vector}$^\circ$'
    real_space.plot(real_space_title)

    # Generate diffraction patterns in 2D and 1D of real space
    diffraction_pattern_of_real_space = DiffractionPattern(real_space, wavelength, pixel_size, npt)
    diffraction_pattern_title = f'2D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
    diffraction_pattern_of_real_space.plot_2d(diffraction_pattern_title, clim=1e8)
    diff_1D_title = f'1D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
    diffraction_pattern_of_real_space.plot_1d(diff_1D_title)
    plt.show()

    filename = f'calamitic_p{particle_length}x{particle_width}_uv{unit_vector}'
    diffraction_pattern_of_real_space.save_1d(filename)
