"""
Correlation analysis of a (simulated) 2D diffraction pattern
Author: Michael Hassett (Original code by Andrew Martin)
Created: 2023-12-11, copied from pypadf/fxstools/correlationTools.py
"""
import numpy as np
import scipy.ndimage as sdn
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from diffraction import Diffraction2D
from utils import timer, save


class PolarAngularCorrelation:
    """Contains useful methods for calculating angular correlations
    """

    @timer
    def __init__(self, diffraction_object: Diffraction2D, num_r, num_th, r_min=0, r_max=None, th_min=0, th_max=360,
                 *,
                 q_instead: bool = False, subtract_mean: bool = False, real_only: bool = False):
        """Converting a 2D diffraction image into an r v. theta plot
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

        self._polar_plot = None
        self._fig_polar = None
        self._ax_polar = None
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

        self.ang_corr = None
        self.__fig_corr__ = None
        self.__ax_corr__ = None
        self.__fig_corr_point__ = None
        self.__ax_corr_point__ = None

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
        self._polar_plot = data.reshape(num_r, num_th)
        if subtract_mean:
            self.subtract_mean_r()
        if real_only:
            self._polar_plot = np.real(self._polar_plot)

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

    def plot(self, title=None, clim=None):
        self._fig_polar, self._ax_polar = plt.subplots()
        plot = self._ax_polar.imshow(self._polar_plot, aspect='auto')
        self._ax_polar.invert_yaxis()
        self._ax_polar.set_xlabel('$\Theta$ / $^\circ$')
        if self.q_instead:
            self._ax_polar.set_ylabel('q')
        else:
            self._ax_polar.set_ylabel('r')
        self._ax_polar.set_xticks(np.arange(0, self.num_th, (self.num_th / self.th_max) * 45),
                                  np.arange(self.th_min, self.th_max, 45))
        if title is not None:
            self._ax_polar.set_title(title)
        self._fig_polar.tight_layout()
        divider = make_axes_locatable(self._ax_polar)

        # creating new axes on the right side of current axes(ax).
        # The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
        colorbar_axes = divider.append_axes("right",
                                            size="10%",
                                            pad=0.1)
        self._fig_polar.colorbar(plot, cax=colorbar_axes)
        if clim is not None:
            plot.set_clim(0, clim)

    def save(self, file_name, file_type=None, **kwargs):
        """
        Save the polar plot as a numpy file or image file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self._fig_polar, self._polar_plot, file_name, file_type, **kwargs)
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
        av = np.average(self._polar_plot, 1)
        self._polar_plot -= np.outer(av, np.ones(self._polar_plot.shape[1]))

    # performs the angular correlation of each q-shell with itself
    def angular_correlation(self, polar2=None):
        """
        Calculate the 2D angular correlation of a polar plot
        or cross-correlation of two polar plots

        Parameters
        ----------
        polar2 : numpy array (float)
            second input polar plot.
            If provided, then cross-correlation is computed
            between polar and polar2.
            If polar2 not provided, then auto-correlation of
            polar is computed

        Returns
        -------
        out : numpy array (float)
            angular correlation function
        """

        fpolar = np.fft.fft(self._polar_plot, axis=1)

        if polar2 is not None:
            fpolar2 = np.fft.fft(polar2, axis=1)
            corr = np.fft.ifft(fpolar2.conjugate() * fpolar, axis=1)
        else:
            corr = np.fft.ifft(fpolar.conjugate() * fpolar, axis=1)
        self.ang_corr = corr
        return self.ang_corr

    def plot_angular_correlation(self, title=None, clim=None):
        print(f'Plotting full angular correlation...')
        self.__fig_corr__, self.__ax_corr__ = plt.subplots()
        plot = self.__ax_corr__.imshow(np.real(self.ang_corr), aspect='auto')
        self.__ax_corr__.invert_yaxis()
        if title is not None:
            self.__ax_corr__.set_title(title)
        self.__ax_corr__.set_xlabel('$\Theta$ / $^\circ$')
        if self.q_instead:
            self.__ax_corr__.set_ylabel('q')
        else:
            self.__ax_corr__.set_ylabel('r')
        self.__ax_corr__.set_xticks(np.arange(0, self.num_th, (self.num_th / self.th_max) * 45),
                                    np.arange(self.th_min, self.th_max, 45))

        self.__fig_corr__.tight_layout()
        divider = make_axes_locatable(self.__ax_corr__)

        # creating new axes on the right side of current axes(ax).
        # The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
        colorbar_axes = divider.append_axes("right",
                                            size="10%",
                                            pad=0.1)
        self.__fig_corr__.colorbar(plot, cax=colorbar_axes)
        if clim:
            plot.set_clim(0, clim)

    def save_angular_correlation(self, file_name, file_type='jpeg', **kwargs):
        """
        Save the angular correlation as a numpy file or image file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_corr__, self.ang_corr, file_name, file_type, **kwargs)
        print(f'Saved angular correlation as {file_name}')

    def plot_angular_correlation_point(self, point: float, title=None, y_lim=None, *,
                                       save_fig: bool = False, save_name: str = None,
                                       save_type: str = 'jpeg', **kwargs):
        """
        Plot the angular correlation at a point
        :param point: r (or q) that you wish to plot
        :param title: title for the plot
        :param y_lim: top limit of the y axis
        :param save_fig: Whether to save the figure (default: False)
        :param save_name: File name to save under
        :param save_type: File type to save as (e.g. png or jpg). Default jpeg file
        :param kwargs: Any keyword arguments to pass to matplotlib.pyplot.savefig
        :return:
        """
        print(f'Plotting angular correlation at {point}...')
        self.__fig_corr_point__, self.__ax_corr_point__ = plt.subplots()
        self.__ax_corr_point__.plot(np.real(self.ang_corr[point, :]))
        if title is not None:
            self.__ax_corr_point__.set_title(title)
        self.__ax_corr_point__.set_xlabel('$\Theta$ / $^\circ$')
        self.__ax_corr_point__.set_ylabel('Intensity (arb. units)')
        self.__ax_corr_point__.set_xticks(np.arange(0, self.num_th, (self.num_th / self.th_max) * 45),
                                          np.arange(self.th_min, self.th_max, 45))
        self.__fig_corr_point__.tight_layout()
        self.__ax_corr_point__.set_xlim(0, self.num_th / 2)
        if y_lim:
            self.__ax_corr_point__.set_ylim(y_lim[0], y_lim[1])
        if save_fig:
            self.save_angular_correlation_point(save_name, save_type, **kwargs)

    def save_angular_correlation_point(self, file_name, file_type=None, **kwargs):
        """
        Save the angular correlation as a numpy file or image file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_corr_point__, self.ang_corr, file_name, file_type, **kwargs)
        print(f'Saved angular correlation as {file_name}')

    # angular correlation of each q-shell with all other q-shells
    #
    def angular_intershell_correlation(self, polar, polar2=None, real_only=True):
        """
        Calculate the 3D angular correlation [C(q,q',theta)] of a polar plot
        or cross-correlation of two polar plots.
        Each q-ring is correlated with every other q-ring.

        Parameters
        ----------
        polar : numpy array (float)
            input polar plot

        polar2 : numpy array (float)
            second input polar plot.
            If provided, then cross-correlation is computed
            between polar and polar2.
            If polar2 not provided, then auto-correlation of
            polar is computed

        real_only : bool
            enure output if real valued if True.

        Returns
        -------
        out : numpy array (float)
            angular correlation function
        """

        fpolar = np.fft.fft(polar, axis=1)

        if np.any(polar2) != None:
            fpolar2 = np.fft.fft(polar2, axis=1)
        else:
            fpolar2 = fpolar

        out = np.zeros((polar.shape[0], polar.shape[0], polar.shape[1]), dtype=np.complex128)
        for i in np.arange(polar.shape[0]):
            for j in np.arange(polar.shape[0]):
                out[i, j, :] = fpolar[i, :] * fpolar2[j, :].conjugate()
        out = np.fft.ifft(out, axis=2)

        if real_only:
            out = np.real(out)
        return out

    def apply_mask(self, func, mask):
        """
        Multiplies an array by a mask array

        Parameters
        ----------
        func : numpy array
            numpy array of data

        mask : numpy array
            mask array containing 0s and 1s

        Returns
        -------
        func*mask : numpy array
        """
        return func * mask

    def mask_correction(self, corr, maskcorr):
        """
        Corrected correlation function for effects of the mask.
        Divides corr by maskcorr wherever maskcorr is greater than 0.

        Parameters
        ----------
        corr : numpy array
            correlation function

        maskcorr : numpy array
            correlation of mask function

        Returns
        -------
        corr : numpy array
            correlation data divided by mask correlation
        """
        imask = np.where(maskcorr != 0)
        corr[imask] *= 1.0 / maskcorr[imask]
        return corr

    #
    # pairwise correlation of (flattened) arrays
    #
    # not for angular correlations; good for correlation of mean asic values
    #
    def allpixel_correlation(self, arr1, arr2):
        """
        Returns the outer product between the flattened
        arr1 and arr2.
        """
        out = np.outer(arr1.flatten(), arr2.flatten())
        return out

    def pearsonCorrelation_2D(self, arr1, arr2, *, lim=None, angular: bool = False):
        """
        Computes the Pearson correlation between two polar plots
        as a function of radial q-bin.

        Parameters
        ----------
        arr1, arr2 : numpy arrays (floats)
            Arrays with the same number of elements

        Returns
        -------
        pc : numpy array
            Pearson correlation values as a function of q (1D)
        """
        if lim is None:
            lim = [0, arr1.shape[0], 0, arr1.shape[1]]
        a1 = arr1[lim[0]:lim[1], lim[2]:lim[3]]
        a2 = arr2[lim[0]:lim[1], lim[2]:lim[3]]

        if angular:
            c1 = a1 - np.outer(np.average(a1, 1), np.ones(a1.shape[1]))
            c2 = a2 - np.outer(np.average(a2, 1), np.ones(a2.shape[1]))
            pc = np.sum(c1 * c2, 1) / np.sqrt(np.sum(c1 * c1, 1) * np.sum(c2 * c2, 1))
        else:
            c1 = a1 - np.average(a1)
            c2 = a2 - np.average(a2)
            pc = np.sum(c1 * c2) / np.sqrt(np.sum(c1 * c1) * np.sum(c2 * c2))
        return pc
