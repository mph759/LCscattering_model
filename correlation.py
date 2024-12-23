"""
Correlation analysis of a (simulated) 2D diffraction pattern
Author: Michael Hassett (Original code by Andrew Martin)
Created: 2023-12-11, copied from pypadf/fxstools/correlationTools.py
"""
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Callable

from diffraction import PolarDiffraction2D, Diffraction2D
from utils import timer, save, ParameterReader, align_ylim


class AngularCorrelation:
    @timer
    def __init__(self, diffraction_2d_polar: PolarDiffraction2D) -> None:
        """
        Performing angular correlation analysis on polar diffraction pattern
        :param diffraction_2d_polar: PolarDiffraction2D object
        """
        self.polar_diffraction = diffraction_2d_polar.data
        self.num_r = diffraction_2d_polar.num_r
        self.r_min = diffraction_2d_polar.r_min
        self.r_max = diffraction_2d_polar.r_max
        self.num_th = diffraction_2d_polar.num_th
        self.th_min = diffraction_2d_polar.th_min
        self.th_max = diffraction_2d_polar.th_max
        self.q_instead = diffraction_2d_polar.q_instead
        self.ang_corr = self.angular_correlation()
        self.__fig_corr__ = None
        self.__ax_corr__ = None
        self.__fig_corr_point__ = None
        self.__ax_corr_point__ = None

    def angular_correlation(self, polar2=None) -> np.ndarray:
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

        fpolar = np.fft.fft(self.polar_diffraction, axis=1)

        if polar2 is not None:
            fpolar2 = np.fft.fft(polar2, axis=1)
            corr = np.fft.ifft(fpolar2.conjugate() * fpolar, axis=1)
        else:
            corr = np.fft.ifft(fpolar.conjugate() * fpolar, axis=1)
        return corr

    @classmethod
    def load(cls, file_dir: Path | str, num_r=None, num_th=None, r_min=0, r_max=None, th_min=0, th_max=360,
             *, q_instead: bool = False, reader: ParameterReader | None = None) -> 'AngularCorrelation':
        if num_r is None and reader is None:
            try:
                reader = ParameterReader(file_dir)
            except FileNotFoundError as e:
                print(f'Could not find file parameter file automatically at {file_dir}')
                reader = None
        if reader is not None:
            params = reader.params['Polar 2D Diffraction']
            num_r = params['num_r']
            r_min = params['r_min']
            r_max = params['r_max']
            num_th = params['num_th']
            th_min = params['th_min']
            th_max = params['th_max']
            q_instead = params['q_instead']

        numpy_file = Path(file_dir) / 'angular_corr.npy'
        if num_r is None or num_th is None:
            raise AttributeError('num_r and num_th must be provided, or a ParameterReader object must be provided')
        new_correlation = cls.__new__(cls)
        new_correlation.polar_diffraction = np.empty((num_r, num_th))
        new_correlation.num_r = num_r
        new_correlation.r_min = r_min
        new_correlation.r_max = r_max
        new_correlation.num_th = num_th
        new_correlation.th_min = th_min
        new_correlation.th_max = th_max
        new_correlation.q_instead = q_instead
        new_correlation.ang_corr = np.load(numpy_file)
        new_correlation.__ax_corr__ = None
        new_correlation.__fig_corr__ = None
        new_correlation.__ax_corr_point__ = None
        new_correlation.__fig_corr_point__ = None
        return new_correlation

    def plot(self, title=None, clim=None, fig: plt.Figure | None = None, ax: plt.Axes | None = None):
        print(f'Plotting full angular correlation...')
        if ax is None:
            self.__fig_corr__, self.__ax_corr__ = plt.subplots()
        else:
            self.__fig_corr__, self.__ax_corr__ = fig, ax
        plot = self.__ax_corr__.imshow(np.real(self.ang_corr), aspect='auto')
        self.__ax_corr__.invert_yaxis()
        if title is not None:
            self.__ax_corr__.set_title(title)
        self.__ax_corr__.set_xlabel('$\Theta$ / $^\circ$')
        if self.q_instead:
            self.__ax_corr__.set_ylabel('q')
        else:
            self.__ax_corr__.set_ylabel('r')
        self.__ax_corr__.set_xticks(np.arange(0, self.num_th, (
                self.num_th / self.th_max) * 45),
                                    np.arange(self.th_min, self.th_max, 45))

        # creating new axes on the right side of current axes(ax).
        # The width of cax will be 5% of ax and the padding between cax and ax will be fixed at 0.05 inch.
        colorbar_axes = make_axes_locatable(self.__ax_corr__).append_axes("right", size="5%", pad=0.1)
        self.__fig_corr__.colorbar(plot, cax=colorbar_axes)
        if clim:
            plot.set_clim(0, clim)
        self.__fig_corr__.tight_layout()

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
                  func: Optional[Callable] = None,
                  fig: plt.Figure | None = None, ax: plt.Axes | None = None, step: float = 0, label: str | None = None,
                  save_fig: bool = False, save_name: str = None, save_type: str = 'png', **kwargs):
        """
        Plot the angular correlation at a point
        :param func:
        :param point: r (or q) that you wish to plot
        :param title: title for the plot
        :param y_lim: max and minimum limit of the y-axis (as a tuple). If None (default) will auto-scale
        :param fig: matplotlib Figure
        :param ax: matplotlib Axes
        :param step: step size between plots (for plotting multiple lines)
        :param label: label for the plot (for a legend)
        :param save_fig: Whether to save the figure (default: False)
        :param save_name: File name to save under
        :param save_type: File type to save as (e.g. png or jpg). Default png file
        :param kwargs: Any keyword arguments to pass to matplotlib.pyplot.savefig
        :return:
        """
        print(f'Plotting angular correlation at {point}...')
        array = np.real(self.ang_corr[point, :]) + step
        if func is not None:
            array = func(array)
        if ax is None:
            self.__fig_corr_point__, self.__ax_corr_point__ = plt.subplots()
        else:
            self.__fig_corr_point__, self.__ax_corr_point__ = fig, ax
        if label is None:
            self.__ax_corr_point__.plot(np.linspace(0, 360, self.num_th), array)
        else:
            self.__ax_corr_point__.plot(np.linspace(0, 360, self.num_th), array, label=label)
        if title is not None:
            self.__ax_corr_point__.set_title(title)
        self.__ax_corr_point__.set_xlabel('$\Theta$ / $^\circ$')
        self.__ax_corr_point__.set_ylabel('Intensity (arb. units)')

        self.__ax_corr_point__.set_xlim(0, 180)
        if y_lim is not None:
            self.__ax_corr_point__.set_ylim(y_lim[0], y_lim[1])
        else:
            align_ylim(self.__ax_corr_point__, x_range=(0,180), edge_mask=2, scale=1.5)
        self.__fig_corr_point__.tight_layout()
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
        fpolar = np.fft.fft(self.polar_diffraction, axis=1)

        if np.any(polar2):
            fpolar2 = np.fft.fft(polar2, axis=1)
        else:
            fpolar2 = fpolar

        out = np.zeros(
            (self.polar_diffraction.shape[0], self.polar_diffraction.shape[0], self.polar_diffraction.shape[1]),
            dtype=np.complex128)
        for i in np.arange(self.polar_diffraction.shape[0]):
            for j in np.arange(self.polar_diffraction.shape[0]):
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
