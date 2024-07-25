"""
Correlation analysis of a (simulated) 2D diffraction pattern
Author: Michael Hassett (Original code by Andrew Martin)
Created: 2023-12-11, copied from pypadf/fxstools/correlationTools.py
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from diffraction import PolarDiffraction2D
from utils import timer, save


class AngularCorrelation:
    def __init__(self, diffraction_2d_polar: PolarDiffraction2D):
        self._polar_plot = diffraction_2d_polar.polar_plot
        self.angular_correlation()
        self.__fig_corr__ = None
        self.__ax_corr__ = None
        self.__fig_corr_point__ = None
        self.__ax_corr_point__ = None

    @property
    def polar(self):
        return self._polar_plot

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

    def plot(self, title=None, clim=None):
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

    def save(self, file_name, file_type='png', **kwargs):
        """
        Save the angular correlation as a numpy file or image file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_corr__, self.ang_corr, file_name, file_type, **kwargs)
        print(f'Saved angular correlation as {file_name}')

    def plot_line(self, point: float, title=None, y_lim: tuple[float, float] = None, *,
                  save_fig: bool = False, save_name: str = None,
                  save_type: str = 'png', **kwargs):
        """
        Plot the angular correlation at a point
        :param point: r (or q) that you wish to plot
        :param title: title for the plot
        :param y_lim: max and minimum limit of the y-axis (as a tuple). If None (default) will auto-scale
        :param save_fig: Whether to save the figure (default: False)
        :param save_name: File name to save under
        :param save_type: File type to save as (e.g. png or jpg). Default png file
        :param kwargs: Any keyword arguments to pass to matplotlib.pyplot.savefig
        :return:
        """
        print(f'Plotting angular correlation at {point}...')
        array = np.real(self.ang_corr[point, :])
        self.__fig_corr_point__, self.__ax_corr_point__ = plt.subplots()
        self.__ax_corr_point__.plot(array)
        if title is not None:
            self.__ax_corr_point__.set_title(title)
        self.__ax_corr_point__.set_xlabel('$\Theta$ / $^\circ$')
        self.__ax_corr_point__.set_ylabel('Intensity (arb. units)')
        self.__ax_corr_point__.set_xticks(np.arange(0, self.num_th, (self.num_th / self.th_max) * 45),
                                          np.arange(self.th_min, self.th_max, 45))
        self.__fig_corr_point__.tight_layout()
        self.__ax_corr_point__.set_xlim(0, self.num_th / 2)
        if y_lim is not None:
            self.__ax_corr_point__.set_ylim(y_lim[0], y_lim[1])
        else:
            edge_mask = 2
            scale = 1.5
            array_cut = array[edge_mask:(self.num_th // 2) - edge_mask]
            y_min, y_max = scale * np.min(array_cut), scale * np.max(array_cut)
            self.__ax_corr_point__.set_ylim(y_min, y_max)
        if save_fig:
            self.save_line(array, save_name, save_type, **kwargs)

    def save_line(self, array, file_name, file_type=None, **kwargs):
        """
        Save the angular correlation as a numpy file or image file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig_corr_point__, array, file_name, file_type, **kwargs)
        print(f'Saved angular correlation as {file_name}')

    # angular correlation of each q-shell with all other q-shells
    #
    def intershell(self, polar2=None, real_only=True):
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
        fpolar = np.fft.fft(self.polar, axis=1)

        if np.any(polar2) != None:
            fpolar2 = np.fft.fft(polar2, axis=1)
        else:
            fpolar2 = fpolar

        out = np.zeros((self.polar.shape[0], self.polar.shape[0], self.polar.shape[1]), dtype=np.complex128)
        for i in np.arange(self.polar.shape[0]):
            for j in np.arange(self.polar.shape[0]):
                out[i, j, :] = fpolar[i, :] * fpolar2[j, :].conjugate()
        out = np.fft.ifft(out, axis=2)

        if real_only:
            out = np.real(out)
        return out

def apply_mask(func, mask):
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

def mask_correction(corr, maskcorr):
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
def allpixel_correlation(arr1, arr2):
    """
    Returns the outer product between the flattened
    arr1 and arr2.
    """
    out = np.outer(arr1.flatten(), arr2.flatten())
    return out

def pearsonCorrelation_2D(arr1, arr2, *, lim=None, angular: bool = False):
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
