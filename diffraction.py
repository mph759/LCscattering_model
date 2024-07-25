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
import scipy.ndimage as sdn
from scipy import signal

from spatial import RealSpace
from utils import timer, save


class Diffraction2D:
    @timer
    def __init__(self,
                 space_object: RealSpace,
                 wavelength: float,
                 pixel_size: float,
                 dx: float,
                 npt: int = 2250,
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

        # creating new axes on the right side of current axes(ax).
        # The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
        colorbar_axes = make_axes_locatable(self.__ax_2d__).append_axes("right", size="5%", pad=0.1)
        self.__fig_2d__.colorbar(plot, cax=colorbar_axes)
        if clim:
            plot.set_clim(0, clim)

        self.__fig_2d__.tight_layout()

    def save(self, file_name, file_type=None, **kwargs):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_2d__, self.pattern_2d, file_name, file_type, **kwargs)
        print(f'Saved 2D diffraction pattern as {file_name}')


class PolarDiffraction2D:
    @timer
    def __init__(self, diffraction_object: Diffraction2D, num_r, num_th, r_min=0, r_max=None, th_min=0, th_max=360,
                 *, q_instead: bool = False, real_only: bool = False):
        """Converting a 2D diffraction image into a polar (r v. theta) plot
        :param num_r: number of radial bins
        :type num_r: int
        :param num_th: number of angular bins
        :type num_th: int
        :param r_min: value of smallest radial bin. (arbitrary units)
        :type r_min: float
        :param r_max: value of largest radial bin
        :type r_max: float
        :param th_min: value of smallest angular bin (radians)
        :type th_min: float
        :param th_max: value of largest angular bin (radians
        :type th_max: float
        :param q_instead: Whether to use q bins instead of r (default: False)
        :param subtract_mean: Whether to subtract mean from the radial ring (default: False)
        :type subtract_mean: bool
        :param real_only: Whether to return only the real component of the array (default: False)
        :type real_only: bool
        :return Data interpolated onto an r vs theta grid
        :rtype numpy array (float)
        """
        self._data_2d = diffraction_object.pattern_2d
        self._centre = tuple([dimension / 2 for dimension in diffraction_object.space.grid])
        self._pixel_size = diffraction_object.pixel_size
        self._wavelength = diffraction_object.wavelength
        self._detector_dist = diffraction_object.detector_dist

        self._fig = None
        self._ax = None
        self.num_r = num_r
        self.num_th = num_th
        self.r_min = r_min
        if r_max is None:
            self.r_max = num_r
        else:
            self.r_max = r_max
        self.th_min = th_min
        self.th_max = th_max
        self.th_min_rad, self.th_max_rad = map(np.deg2rad, (th_min, th_max))
        self.q_instead = q_instead

        if q_instead is not None:
            self.q_bins = self.q_bins(self.num_r)
            num_r = self.q_bins.size
            r_array = np.outer(self.q_bins, np.ones(num_th))
        else:
            r_array = np.outer(np.arange(num_r) * (r_max - r_min) / float(num_r) + r_min, np.ones(num_th))
        th_array = np.outer(np.ones(num_r),
                            np.arange(num_th) * (self.th_max_rad - self.th_min_rad) /
                            float(num_th) + self.th_min_rad)

        new_x = r_array * np.cos(th_array) + self.centre_x
        new_y = r_array * np.sin(th_array) + self.centre_y

        data = sdn.map_coordinates(self.data_2d, [new_x.flatten(), new_y.flatten()], order=3)
        self._data = data.reshape(num_r, num_th)
        if real_only:
            self._data = np.real(self._data)

    @property
    def params(self):
        return ('Polar Angular Correlation',
                {'num_r': self.num_r,
                 'num_th': self.num_th,
                 'r_min': self.r_min,
                 'r_max': self.r_max,
                 'th_min': self.th_min,
                 'th_max': self.th_max,
                 'q_instead': self.q_instead})

    @property
    def data_2d(self):
        return self._data_2d

    @property
    def data(self):
        return self._data

    @property
    def centre_x(self):
        return self._centre[0]

    @property
    def centre_y(self):
        return self._centre[1]

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def detector_dist(self):
        return self._detector_dist

    def gaussian_convolve(self, side_length: int = 3, stddev: int = 1):
        kernel = [signal.windows.gaussian(side_length, stddev) for _ in range(side_length)]
        self._data = signal.fftconvolve(self.data, kernel, mode='same')
        return kernel

    def plot(self, title=None, clim=None):
        self._fig, self._ax = plt.subplots()
        plot = self._ax.imshow(self.data, aspect='auto')
        self._ax.invert_yaxis()
        self._ax.set_xlabel('$\Theta$ / $^\circ$')
        if self.q_instead:
            self._ax.set_ylabel('q')
        else:
            self._ax.set_ylabel('r')
        self._ax.set_xticks(np.arange(0, self.num_th, (self.num_th / self.th_max) * 45),
                            np.arange(self.th_min, self.th_max, 45))
        if title is not None:
            self._ax.set_title(title)
        self._fig.tight_layout()
        divider = make_axes_locatable(self._ax)

        # creating new axes on the right side of current axes(ax).
        # The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
        colorbar_axes = divider.append_axes("right",
                                            size="5%",
                                            pad=0.1)
        self._fig.colorbar(plot, cax=colorbar_axes)
        if clim is not None:
            plot.set_clim(0, clim)

    def save(self, file_name, file_type=None, **kwargs):
        """
        Save the polar plot as a numpy file or image file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self._fig, self._data, file_name, file_type, **kwargs)
        print(f'Saved polar plot as {file_name}')

    def q_bins(self, nq):
        """
        Generates a list of q bins based on Ewald sphere curvature
        :param nq: number of q bins
        :type nq: int
        :return: list of q values of each radial (q) bin
        :rtype: numpy array (float)
        """
        pixel_max = self.centre_x
        q_max = (2 / self.wavelength) * np.sin(
            np.arctan(pixel_max * self.pixel_size / self.detector_dist) / 2.0)
        q_ind = np.arange(nq) * q_max / float(nq)
        q_pixels = (self.detector_dist / self.pixel_size) * np.tan(
            2.0 * np.arcsin(q_ind * (self.wavelength / 2.0)))
        return np.floor(q_pixels)

    def subtract_mean_r(self):
        """
        Subtract the mean value in each q-ring from a polar plot

        Parameters
        ----------
        pplot : numpy array (float)
            input polar plot

        Returns
        -------
        out : numpy array (float)
            polar plot with q-ring mean value subtracted
        """
        av = np.average(self._data, 1)
        self._data -= np.outer(av, np.ones(self._data.shape[1]))


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
                    pixelX=self.dx, pixelY=self.dx)
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
