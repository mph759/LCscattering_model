"""
Generating 2D and 1D diffraction patterns from an arrangement of particles in real space
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from scipy.ndimage import rotate

from spatial import RealSpace
from utils import timer, save, gaussian_convolve


class Diffraction2D:
    @timer
    def __init__(self, space_object: RealSpace, wavelength: float, pixel_size: float, dx: float, npt: int = 2250,
                 rotation: float = None):
        """
        Generating 2D diffraction patterns on a real space object
        :param space_object: RealSpace object with particles to be diffracted
        :param wavelength: Wavelength of the beam
        :param pixel_size: Size of the pixels on the simulated detector
        :param npt: Number of points in radial dimension for radial integration
        :param dx: Ratio of real length in metres to number of pixels
        :param rotation: Rotation of the diffraction pattern in degrees
        """

        self.space = space_object
        self._pattern_2d = self.create_2d_diffraction()
        self.wavelength = wavelength
        self._pixel_size = pixel_size
        self._dx = dx
        self._npt = npt
        self._num_pixels = self.space.grid[0]
        # self.detector_dist = self.pixel_size * self.space.grid[0] / self.wavelength
        self._detector_dist = ((self.num_pixels * self.pixel_size) /
                               (2 * np.tan(2 * np.arcsin(self.wavelength / (4 * self.dx)))))
        if rotation is not None:
            self.rotate_image(rotation)
        # Initialise plotting objects
        self.__fig_2d__ = None
        self.__ax_2d__ = None

    @property
    def params(self):
        return ("Diffraction",
                {'wavelength': self.wavelength,
                 'detector_dist': self.detector_dist,
                 'num_pixels': self.num_pixels,
                 'pixel_size': self.pixel_size,
                 'dx': self.dx,
                 'npt': self.npt})

    @property
    def npt(self):
        return self._npt

    @property
    def pattern_2d(self):
        return self._pattern_2d

    @property
    def num_pixels(self):
        return self._num_pixels

    @property
    def dx(self):
        return self._dx

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def detector_dist(self):
        return self._detector_dist

    def __add__(self, other):
        if isinstance(other, Diffraction2D):
            if self.params == other.params:
                self._pattern_2d += other.pattern_2d
                return self
            else:
                raise ValueError(f"{self} and {other} do not share the same parameters")
        else:
            return TypeError(f'unsupported operand type(s) for +: \'{type(self)}\' and \'{type(other)}\'')

    def __truediv__(self, other):
        if isinstance(other, int):
            self._pattern_2d = np.divide(self.pattern_2d, other)
            return self
        else:
            return TypeError(f'unsupported operand type(s) for +: \'{type(self)}\' and \'{type(other)}\'')

    def create_2d_diffraction(self):
        """
        Simulating the 2D diffraction from the given real space object
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

    def rotate_image(self, rotation):
        self._pattern_2d = rotate(self._pattern_2d, angle=rotation, reshape=False)

    def gaussian_convolve(self, length: int = None, stddev: int = None):
        self._pattern_2d = gaussian_convolve(self.pattern_2d, length, stddev)

    def plot(self, title, clim: float = None, peaks: list[int] = None):
        """
        Plot the 2D Diffraction image
        :param title: String to be placed as a title on the figure
        :param clim: Colour bar limit
        :param peaks: List of peaks to plot as rings on the diffraction pattern
        :return:
        """
        print("Plotting 2D diffraction figure...")
        # Plot the diffraction image
        self.__fig_2d__, self.__ax_2d__ = plt.subplots()
        plot = self.__ax_2d__.imshow(self.pattern_2d ** 2)
        self.__ax_2d__.invert_yaxis()
        self.__ax_2d__.set_title(title)

        if peaks is not None:
            for peak in peaks:
                ring = plt.Circle((self.num_pixels // 2, self.num_pixels // 2), peak, color='r', fill=False, alpha=0.25)
                self.__ax_2d__.add_patch(ring)

        self.__ax_2d__.set_xticks([])
        self.__ax_2d__.set_yticks([])
        self.__fig_2d__.tight_layout()
        divider = make_axes_locatable(self.__ax_2d__)

        # creating new axes on the right side of current axes(ax).
        # The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
        colorbar_axes = divider.append_axes("right",
                                            size="10%",
                                            pad=0.1)
        self.__fig_2d__.colorbar(plot, cax=colorbar_axes)
        if clim:
            plot.set_clim(0, clim)

    def save(self, file_name, file_type=None, **kwargs):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_2d__, self.pattern_2d, file_name, file_type, **kwargs)
        print(f'Saved 2D diffraction pattern as {file_name}')


class Diffraction1D:
    def __init__(self, diffraction_2d):
        self.pattern_2d = diffraction_2d.pattern_2d
        self.space = diffraction_2d.space
        self.wavelength = diffraction_2d.wavelength
        self._pixel_size = diffraction_2d.pixel_size
        self._dx = diffraction_2d.dx
        self._npt = diffraction_2d.npt
        self._num_pixels = diffraction_2d.num_pixels
        # self.detector_dist = self.pixel_size * self.space.grid[0] / self.wavelength
        self._detector_dist = ((self._num_pixels * self.pixel_size) /
                               (2 * np.tan(2 * np.arcsin(self.wavelength / (4 * self.dx)))))
        self._pattern_1d = None
        self.create_1d_diffraction()
        # Initialise plotting objects
        self.__fig_1d__ = None
        self.__ax_1d__ = None

    @property
    def params(self):
        return ("Diffraction",
                {'wavelength': self.wavelength,
                 'detector_dist': self.detector_dist,
                 'pixel_size': self.pixel_size,
                 'dx': self.dx,
                 'npt': self.npt})

    @property
    def npt(self):
        return self._npt

    @property
    def pattern_1d(self):
        return self._pattern_1d

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def dx(self):
        return self._dx

    @property
    def detector_dist(self):
        return self._detector_dist

    def __add__(self, other):
        if isinstance(other, Diffraction1D):
            if self.params == other.params:
                self._pattern_1d += other.pattern_1d
                return self
            else:
                raise ValueError(f"{self} and {other} do not share the same parameters")
        else:
            return TypeError(f'unsupported operand type(s) for +: \'{type(self)}\' and \'{type(other)}\'')

    def __truediv__(self, other):
        if isinstance(other, int):
            self._pattern_2d = np.divide(self.pattern_1d, other)
            return self
        else:
            return TypeError(f'unsupported operand type(s) for +: \'{type(self)}\' and \'{type(other)}\'')

    def radial_integration(self, frame, unit="q_nm^-1"):
        """
        Perform azimuthal integration of frame array
        :param frame: numpy array containing 2D intensity
        :param unit: Unit used in the radial integration
        :return: two-col array of q & intensity.
        """
        # print("Debug - ", self.detector_dist, self.pixel_size, self.wavelength)

        image_center = [dimension / 2 for dimension in self.space.grid]
        ai = AzimuthalIntegrator()
        ai.setFit2D(directDist=self.detector_dist / 1000,
                    centerX=image_center[0],
                    centerY=image_center[1],
                    pixelX=self.pixel_size, pixelY=self.pixel_size)
        ai.wavelength = self.wavelength
        integrated_profile = ai.integrate1d(data=frame, npt=self.npt, unit=unit)
        return np.transpose(np.array(integrated_profile))

    @timer
    def create_1d_diffraction(self) -> None:
        """
        Generating the 1D diffraction from the 2D diffraction image through radial integration
        :return:
        """
        print("Generating 1D diffraction image...")
        radius = self.space.grid[0] // 2
        diffraction_image_cone = self.circular_mask(self.space.grid, radius) * self.pattern_2d
        diffraction_plot = self.radial_integration(diffraction_image_cone, unit="q_nm^-1")

        non_zero = diffraction_plot[:, 1] != 0  # Removes data points at = 0 due to the cone restriction
        self._pattern_1d = diffraction_plot[non_zero]
        print("1D diffraction image complete")

    def __add__(self, other):
        if isinstance(other, Diffraction2D):
            if self.params == other.params:
                self._pattern_1d += other.pattern_1d
                return self
            else:
                raise ValueError(f"{self} and {other} do not share the same parameters")
        else:
            return TypeError(f'unsupported operand type(s) for +: \'{type(self)}\' and \'{type(other)}\'')

    def __truediv__(self, other):
        if isinstance(other, int):
            self._pattern_1d = np.divide(self.pattern_1d, other)
            return self
        else:
            return TypeError(f'unsupported operand type(s) for +: \'{type(self)}\' and \'{type(other)}\'')

    def plot(self, title: str) -> None:
        """
        Plot a 1D diffraction pattern
        :param title: Title text for the plotting
        :return:
        """
        if self.pattern_1d is None:
            self.create_1d_diffraction()
        # Plot 1D integration
        print("Plotting 1D diffraction figure...")
        self.__fig_1d__, self.__ax_1d__ = plt.subplots()
        self.__ax_1d__.plot(self.pattern_1d[int(self.npt // 20):, 0], self.pattern_1d[int(self.npt // 20):, 1])
        self.__ax_1d__.set_title(title)
        self.__ax_1d__.set_xlabel('q / nm$^{-1}$')
        self.__ax_1d__.set_ylabel('Arbitrary Intensity')
        self.__fig_1d__.tight_layout()

    def save(self, file_name, file_type=None, **kwargs) -> None:
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_1d__, self.pattern_1d, file_name, file_type, **kwargs)
        print(f'Saved 1D diffraction pattern as {file_name}')

    @staticmethod
    def circular_mask(grid, mask_radius, show: bool = False):
        """
        Create a circular mask over an image
        :param grid: x and y grid size which the mask will fit over
        :param mask_radius: Radius that the mask should sit on
        :param show: Boolean for whether to show the plot immediately. Default False
        :return:
        """
        kernel = np.zeros(grid)
        filter_y, filter_x = np.ogrid[-mask_radius:mask_radius, -mask_radius:mask_radius]
        mask = filter_x ** 2 + filter_y ** 2 <= mask_radius ** 2
        kernel[mask] = 1
        if show:
            fig, ax = plt.subplots()
            ax.imshow(kernel)
            fig.plot(block=False)
        return kernel
