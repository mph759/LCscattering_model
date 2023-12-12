"""
Author: Andrew Martin, Edited by Michael Hassett
Created: 2023-12-11, copied from pypadf/fxstools/correlationTools.py
"""
from matplotlib import pyplot as plt
from diffraction import DiffractionPattern
import numpy as np
import scipy.ndimage as sdn
from utils import timer


class AngularCorrelation:
    """Contains useful methods for calculating angular correlations
    """

    def __init__(self, diffraction_object: DiffractionPattern):
        self._data_2d = diffraction_object.pattern_2d
        self._centre = tuple([dimension / 2 for dimension in diffraction_object.space.grid])
        self._pixel_size = diffraction_object.pixel_size
        self._wavelength = diffraction_object.wavelength
        self._detector_dist = diffraction_object.detector_dist
        self._polar_plot = None
        self._fig_polar = None
        self._ax_polar = None

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

    @timer
    def polar_plot(self, num_r, num_th, r_min=0, r_max=None, th_min=0, th_max=360, *,
                   q_instead: bool = False, subtract_mean: bool = False, real_only: bool = False, show: bool = False):
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
        :param show: Whether to show the figure (default: False)
        :type show: bool
        :return Data interpolated onto an r vs theta grid
        :rtype numpy array (float)
        """
        th_min_rad, th_max_rad = map(np.deg2rad, (th_min, th_max))
        if r_max is None:
            r_max = num_r
        if q_instead is not None:
            q_bins = self.q_bins(num_r)
            num_r = q_bins.size
            r_array = np.outer(q_bins, np.ones(num_th))
        else:
            r_array = np.outer(np.arange(num_r) * (r_max - r_min) / float(num_r) + r_min, np.ones(num_th))
        th_array = np.outer(np.ones(num_r), np.arange(num_th) * (th_max_rad - th_min_rad) / float(num_th) + th_min_rad)

        new_x = r_array * np.cos(th_array) + self.centre_x
        new_y = r_array * np.sin(th_array) + self.centre_y

        data = sdn.map_coordinates(self._data_2d, [new_x.flatten(), new_y.flatten()], order=3)
        self._polar_plot = data.reshape(num_r, num_th)
        if subtract_mean:
            self.subtract_mean_r()
        if real_only:
            self._polar_plot = np.real(self._polar_plot)
        if show:
            self._fig_polar, self._ax_polar = plt.subplots()
            self._ax_polar.imshow(self._polar_plot)
            self._ax_polar.invert_yaxis()
            self._ax_polar.set_xlabel('$\Theta$ / $^\circ$')
            self._ax_polar.set_ylabel('r')
            self._ax_polar.set_xticks(np.arange(0, num_th, (num_th/th_max)*45), np.arange(th_min, th_max, 45))
            self._ax_polar.set_yticks(np.arange(0, num_r, 100), np.arange(r_min, r_max, 100))

    def q_bins(self, nq):
        """
        Generates a list of q bins based on Ewald sphere curvature
        :param nq: number of q bins
        :type nq: int
        :return: list of q values of each radial (q) bin
        :rtype: numpy array (float)
        """
        pixel_max = self.centre_x
        q_max = (2 / self.wavelength) * np.sin(np.arctan(pixel_max * self.pixel_size / self.detector_dist) / 2.0)
        q_ind = np.arange(nq) * q_max / float(nq)
        q_pixels = (self.detector_dist / self.pixel_size) * np.tan(2.0 * np.arcsin(q_ind * (self.wavelength / 2.0)))
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
    def polarplot_angular_correlation(self, polar2=None):
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
            out = np.fft.ifft(fpolar2.conjugate() * fpolar, axis=1)
        else:
            out = np.fft.ifft(fpolar.conjugate() * fpolar, axis=1)
        return out

    #
    # angular correlation of each q-shell with all other q-shells
    #
    def polarplot_angular_intershell_correlation(self, polar, polar2=None, real_only=True):
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
            Arrays with the same number of elenments

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
