"""
Utilities and other functions
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import inspect
import logging
import os
import sys
import time
import json
import re
from functools import wraps, partial
from pathlib import Path
from stat import S_IREAD
from typing import Any, TypeAlias, Callable, Optional
from astropy.modeling.models import Gaussian1D, Lorentz1D, Voigt1D

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.projections.polar import PolarAxes
import numpy as np
from scipy import signal

Coordinates: TypeAlias = tuple[int, int]


# Specific functions
def generate_positions(space, maximum, change) -> Coordinates:
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spatial oscillations
    :param space: Spacing in x and y-dimensions between positions
    :param maximum: Maximum values in x and y-dimensions
    :param change: Tuple of allowed deviation from initial lattice spacing
    :return: A position in Cartesian coordinates inside the grid
    """
    # Initial positions, just inside the box
    x_space, y_space = space
    x_change, y_change = np.abs(change)
    x_max, y_max = maximum
    x = int(x_space / 2)
    y = int(y_space / 2)

    # Loop while the positions are still inside the box
    while y < y_max:
        x_pos = x
        y_pos = y
        if x_change != 0:
            x_pos += np.random.randint(-x_change, x_change)
        if y_change != 0:
            y_pos += np.random.randint(-y_change, y_change)
        yield x_pos, y_pos
        x += x_space

        # When the position is at the edge of the box, adjust y and reset x
        if x >= x_max:
            y += y_space
            x = x_space


def pythagorean_sides(a: float, b: float, theta: float) -> tuple[float, float]:
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


def plot_angle_bins(samples, mean: float, stddev: float):
    sample_mean = np.mean(samples)
    sample_stddev = np.std(samples)
    sample_size = len(samples)
    bins = range(0, 360, 1)
    fig = plt.figure(figsize=(10, 10))
    counts, bins = np.histogram(samples, bins=bins, density=True)
    ax1 = fig.add_subplot()
    ax1.hist(samples, bins=bins, density=True)
    ax1.set_xlim(0, 360)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    y = 1 / (stddev * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * stddev ** 2))
    for i, angle in enumerate(y):
        if angle < 0:
            angle += 360
        angle %= 360
        y[i] = angle
    ax1.plot(bins, y, 'r')
    ax1.set_xticks(range(bins[0], bins[-1], 45))

    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Angle / \u00B0')

    fig.tight_layout()
    return fig, ax1


def plot_angle_bins_polar(samples, mean: float, stddev: float):
    sample_size = len(samples)
    bins = range(0, 360, 1)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(projection='polar')
    counts, bins = np.histogram(samples, bins=bins, density=True)
    area = counts / sample_size
    radius = (area / np.pi) ** (1 / 2)
    ax.bar(np.radians(bins[:-1]), radius, width=1)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xticks(np.radians(range(bins[0], bins[-1], 15)))
    fig.tight_layout()
    return fig, ax


def chi_squared(samples: np.array, mean: float):
    return np.sum(((samples - mean) ** 2) / mean)


def align_ylim(ax: plt.Axes, x_range=(0, 0), scale: float = 1.5, edge_mask: float = 0):
    line_data = [line.get_data()[1][x_range[0] + edge_mask: x_range[1] - edge_mask] for line in ax.get_lines()]
    min_line = np.min(line_data)
    max_line = np.max(line_data)
    del line_data

    y_min = scale * np.min(min_line)
    y_max = scale * np.max(max_line)
    ax.set_ylim(y_min, y_max)


def init_spacing(particle_length: int, particle_width: int,
                 unit_vector: int, padding_spacing: int) -> tuple[tuple[int, int], tuple[float, float]]:
    """
    Initializes the spacing of the particles based on the particle length and particle width
    :param particle_length:
    :param particle_width:
    :param unit_vector:
    :param padding_spacing:
    :return:
    """
    x_spacing, y_spacing = (spacing + padding
                            for spacing, padding
                            in zip(pythagorean_sides(particle_length, particle_width, unit_vector), padding_spacing))

    # Allow for particles to move slightly in x and y, depending on the spacing
    displacement = tuple([np.ceil(spacing / 2) for spacing in padding_spacing])
    print(f'x spacing: {x_spacing}, y spacing: {y_spacing}')
    print(f'displacement: {displacement}')
    spacing = (x_spacing, y_spacing)
    return spacing, displacement


def subtract_mean(array: np.ndarray, search_override: Optional[Callable] = None) -> np.ndarray:
    if search_override is not None:
        mean = np.mean(search_override(array))
    else:
        mean = np.mean(array)
    return array - mean


def normalize(array: np.ndarray, max_override: Optional[float] = None,
              search_override: Optional[Callable] = None) -> np.ndarray:
    if max_override is not None:
        max_value = max_override
    elif search_override is not None:
        max_value = np.max(search_override(array))
    else:
        max_value = np.max(array)
    return array / max_value


def edge_mask(array: np.ndarray, edge: int = 35) -> np.ndarray:
    edge_index = (len(array) * edge) // 360
    return array[edge_index:-edge_index]


def half_edge_mask(array: np.ndarray, edge: int = 35) -> np.ndarray:
    edge_index = (len(array) * edge) // 360
    return array[edge_index:len(array) // 2 - edge_index]


def convolve_1d(func: Callable) -> np.ndarray:
    @wraps(func)
    def inner(array1: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        array1_fft = np.fft.fft(array1)
        array2_fft = np.fft.fft(func(array1, *args, **kwargs))
        new_array = np.fft.ifft(array1_fft * array2_fft.conjugate())
        return new_array

    return inner


def get_indices_array(array):
    return np.arange(array.shape[0]) - (array.shape[0] // 2)


def triangle(array, *, height: int = 1, width: int = 45):
    indices_array = get_indices_array(array)
    array = (width - np.abs(indices_array)) / width
    array[array < 0] = 0
    return array


def lorentzian(array, *, amplitude: float = 1, x_0: float = 0, fwhm: float = 5):
    indices_array = get_indices_array(array)
    lorentzian_array = Lorentz1D(amplitude=amplitude, x_0=x_0, fwhm=fwhm)
    return lorentzian_array(indices_array)


def voigt(array, *, amplitude: float = 1, x_0: float = 0, fwhm_L: float = 5, fwhm_G: float = 5):
    indices_array = get_indices_array(array)
    voigt_array = Voigt1D(amplitude_L=amplitude, x_0=x_0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
    return voigt_array(indices_array)


def gaussian(array, *, amplitude: float = 1, x_0: float = 0, stddev: float = 5):
    indices_array = get_indices_array(array)
    gaussian_array = Gaussian1D(amplitude=amplitude, mean=x_0, stddev=stddev)
    return gaussian_array(indices_array)


convolve_voigt = convolve_1d(voigt)
convolve_gaussian = convolve_1d(gaussian)


# General functions
def timer(func) -> object:
    """
    Function timer
    :param func: Function to be timed
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        t_start = time.perf_counter()
        result = func(*args, **kwargs)
        t_end = time.perf_counter()
        t_total = t_end - t_start
        func_description = inspect.getdoc(func).split('\n')[0]
        print(f'{func_description} took {t_total:0.4f}s')
        return result

    return wrapper


# Logging parameters
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ParameterLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dir = self.output_dir / 'params.json'
        if not self.dir.exists():
            with open(self.dir, 'x'):
                pass
        self.params = {}

    def __enter__(self) -> 'ParameterLogger':
        self.file = open(self.dir, 'a')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        json.dump(self.params, self.file, cls=NpEncoder, indent=4)
        if self.file:
            self.file.close()
        os.chmod(self.dir, S_IREAD)


class ParameterReader:
    def __init__(self, input_dir: str | Path) -> None:
        self.input_dir = Path(input_dir) / 'params.json'
        if not self.input_dir.exists():
            raise ValueError(f'Input directory {self.input_dir} does not exist')
        with open(self.input_dir, 'r') as f:
            self._params = json.load(f)

    @property
    def params(self) -> dict:
        return self._params


def logger_setup(name, path: str | Path, stream: bool = False, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s - (Line: %(lineno)d [%(filename)s])',
                                  datefmt='%Y/%m/%d %I:%M:%S %p')

    file_path = Path(f'{path}/{name}.log')
    handler = logging.FileHandler(filename=file_path, encoding='utf-8', mode='w')
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)

    return logger


# File handling
def fix_file_ext(file_name: str, file_type: str) -> str:
    """
    Fixes file extension if there is one on the file name which does not match the extension provided
    :param file_name: Potential file name
    :param file_type: Requested file extension
    :return:
    """
    if file_name[-4:] != f".{file_type}":
        file_name = file_name.split('.')[0]
        return f'{file_name}.{file_type}'


def check_existing_ext(file_name: str) -> tuple[str, str]:
    """
    Checks for an existing file extension on a file name
    :param file_name:
    :return: file name and file extension
    """
    file_name_array = file_name.split('.')
    file_name = file_name_array[0]
    if len(file_name_array) > 2:
        raise ValueError("File name cannot include full stop, unless before an extension")
    elif len(file_name_array) > 1:
        file_ext = file_name_array[1]
    else:
        file_ext = None
    return file_name, file_ext


def save(fig: plt.figure, array: np.ndarray, file_name: str, file_type: str = None, close_fig: bool = True,
         **kwargs) -> str:
    """
    Save the figure as a numpy file or as an image
    :param fig: Figure object to be saved
    :param array: numpy array to be saved
    :param file_name: Output file name
    :param file_type: Type of file you want to save (e.g. npy or jpg).
    :param close_fig: Boolean for whether to close the figure after saving.
    If not given, file name is checked for existing extension. Otherwise, default npy file
    :return:
    """
    if file_type is None:
        file_name, file_type = check_existing_ext(file_name)
        if file_type is None:
            file_type = "npy"
    file_name = fix_file_ext(file_name, file_type)
    if file_type == "npy":
        np.save(Path(file_name), array)
    else:
        try:
            fig.savefig(Path(file_name), format=file_type, **kwargs)
        except ValueError:
            raise ValueError(f"Format \'{file_type}\' is not supported (supported formats: npy, eps, jpeg, jpg, pdf, "
                             f"pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp)")
    if close_fig:
        plt.close(fig)
    return file_name


def alphanum_key(s):
    text_list = re.split(r'([_(),\s]+)', str(s))
    for i, text in enumerate(text_list):
        if text.isdigit() or text.lstrip('-').isdigit():
            text_list[i] = int(text)
    return text_list


if __name__ == '__main__':
    mean_angle = 0
    angle_stddev = 2
    angles = np.random.normal(mean_angle, angle_stddev, int(1e6))
    for i, angle in enumerate(angles):
        if angle < 0:
            angle += 360
        angle %= 360
        angles[i] = angle
    fig, ax = plot_angle_bins(angles, mean_angle, angle_stddev)
    fig, ax = plot_angle_bins_polar(angles, mean_angle, angle_stddev)
    plt.show()
