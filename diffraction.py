"""
Generating 2D and 1D diffraction patterns from an arrangement of particles in real space
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import matplotlib.pyplot as plt
import numpy as np
from utils import timer, save
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from spatial import RealSpace


class DiffractionPattern:
    @timer
    def __init__(self, space_object: RealSpace, wavelength: float, pixel_size: float, dx: float, npt: int = 2250):
        """
        Generating 2D diffraction patterns on a real space object
        :param space_object: RealSpace object with particles to be diffracted
        :param wavelength: Wavelength of the beam
        :param pixel_size: Size of the pixels on the simulated detector
        :param npt: Number of points in radial dimension for radial integration
        :param dx: Ratio of real length in metres to  number of pixels
        """

        self.space = space_object
        self._pattern_2d = self.create_2d_diffraction()
        self.wavelength = wavelength
        self._pixel_size = pixel_size
        self._dx = dx
        self._npt = npt
        self._num_pixels = self.space.grid[0]
        # self.detector_dist = self.pixel_size * self.space.grid[0] / self.wavelength
        self._detector_dist = ((self._num_pixels * self.pixel_size) /
                               (2 * np.tan(2 * np.arcsin(self.wavelength / (4 * self._dx)))))
        self._pattern_1d = None
        # Initialise plotting objects
        self.__fig_1d__ = None
        self.__ax_1d__ = None
        self.__fig_2d__ = None
        self.__ax_2d__ = None

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

    @property
    def pattern_2d(self):
        return self._pattern_2d

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def detector_dist(self):
        return self._detector_dist

    def plot_2d(self, title, clim: float = None):
        """
        Plot the 2D Diffraction image
        :param title: String to be placed as a title on the figure
        :param clim: Colour bar limit
        :return:
        """
        print("Plotting 2D diffraction figure...")
        # Plot the diffraction image
        self.__fig_2d__, self.__ax_2d__ = plt.subplots()
        plot = self.__ax_2d__.imshow(self.pattern_2d ** 2)
        self.__ax_2d__.set_title(title)
        self.__fig_2d__.colorbar(plot)
        self.__fig_2d__.tight_layout()
        if clim:
            plot.set_clim(0, clim)

    def save_2d(self, file_name, file_type=None, **kwargs):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_2d__, self.pattern_2d, file_name, file_type, **kwargs)
        print(f'Saved 2D diffraction pattern as {file_name}')

    def frm_integration(self, frame, unit="q_nm^-1"):
        """
        Perform azimuthal integration of frame array
        :param frame: numpy array containing 2D intensity
        :param unit: Unit used in the radial integration
        :return: two-col array of q & intensity.
        """
        # print("Debug - ", self.detector_dist, self.pixel_size, self.wavelength)

        image_center = [dimension / 2 for dimension in self.space.grid]
        ai = AzimuthalIntegrator()
        ai.setFit2D(directDist=self._detector_dist / 1000,
                    centerX=image_center[0],
                    centerY=image_center[1],
                    pixelX=self._pixel_size, pixelY=self._pixel_size)
        ai.wavelength = self.wavelength
        integrated_profile = ai.integrate1d(data=frame, npt=self._npt, unit=unit)
        return np.transpose(np.array(integrated_profile))

    @timer
    def create_1d_diffraction(self):
        """
        Generating the 1D diffraction from the 2D diffraction image through radial integration
        :return:
        """
        print("Generating 1D diffraction image...")
        radius = self.space.grid[0] // 2
        diffraction_image_cone = circular_mask(self.space.grid, radius) * self.pattern_2d
        diffraction_plot = self.frm_integration(diffraction_image_cone, unit="q_nm^-1")

        non_zero = diffraction_plot[:, 1] != 0  # Removes data points at = 0 due to the cone restriction
        diffraction_plot_filtered = diffraction_plot[non_zero]
        print("1D diffraction image complete")
        return diffraction_plot_filtered

    def pattern_1d(self):
        return self._pattern_1d

    def plot_1d(self, title: str):
        """
        Plot a 1D diffraction pattern
        :param title: Title text for the plotting
        :return:
        """
        # Plot 1D integration
        print("Plotting 1D diffraction figure...")
        self.__fig_1d__, self.__ax_1d__ = plt.subplots()
        self.__ax_1d__.plot(self._pattern_1d[int(self._npt // 20):, 0], self._pattern_1d[int(self._npt // 20):, 1])
        self.__ax_1d__.set_title(title)
        self.__ax_1d__.set_xlabel('q / nm$^{-1}$')
        self.__ax_1d__.set_ylabel('Arbitrary Intensity')
        self.__fig_1d__.tight_layout()

    def save_1d(self, file_name, file_type=None, **kwargs):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_1d__, self._pattern_1d, file_name, file_type, **kwargs)
        print(f'Saved 1D diffraction pattern as {file_name}')


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
