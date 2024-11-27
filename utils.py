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
from functools import wraps
from pathlib import Path
from stat import S_IREAD
from typing import Any, TypeAlias, Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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


def plot_angle_bins(samples, mean, stddev, min_x=0, max_x=180, view_table: bool = True):
    sample_mean = np.mean(samples)
    sample_stddev = np.std(samples)
    sample_size = len(samples)
    bins = range(0, 180, 1)
    fig, ax = plt.subplots()
    count, bins, ignored = ax.hist(samples, bins=bins, density=True)
    y = 1 / (stddev * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * stddev ** 2))
    ax.plot(bins, y, 'r')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, 0.09)

    if view_table:
        ax.text(0.7, 0.87, f'Sample size: {sample_size}', transform=plt.gca().transAxes)
        means = [f'{sample_mean:0.2f}', f'{mean:0.2f}']
        stddevs = [f'{sample_stddev:0.2f}', f'{stddev:0.2f}']
        row_labels = ['Mean', 'Stddev']
        col_labels = ['Sample Set', 'Model']
        ax.table(cellText=[means, stddevs], rowLabels=row_labels, colLabels=col_labels,
                 colWidths=[0.1] * 2, bbox=[0.7, 0.75, 0.25, 0.1], transform=plt.gca().transAxes, fontsize=16)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel('Probability')
    ax.set_xlabel('Value')
    fig.tight_layout()
    return fig, ax

def chi_squared(samples: np.array, mean: float):
    return np.sum(((samples - mean) ** 2) / mean)



def align_ylim(ax: plt.Axes, x_range=(0, 0), scale:float = 1.5, edge_mask:float = 0):
    line_data = [line.get_data()[1][x_range[0]+edge_mask: x_range[1] - edge_mask] for line in ax.get_lines()]
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


def gaussian_convolve(array: np.ndarray, length: int = 3, stddev: int = 1) -> np.ndarray:
    """
    Utilise fftconvolve to convolve the array with a Gaussian kernel
    :param array: Array to be "blurred". Can be either 1D or 2D array
    :param length: Side length of the Gaussian kernel
    :param stddev: Standard deviation of Gaussian kernel used to convolve the array
    :return: blurred array
    """
    if length is None:
        length = 3
    if stddev is None:
        stddev = 1
    dim = len(np.shape(array))
    if dim == 2:
        kernel = np.outer(signal.windows.gaussian(length, stddev), signal.windows.gaussian(length, stddev))
    elif dim == 1:
        kernel = signal.windows.gaussian(length, stddev)
    else:
        raise ValueError('The array must be 1D or 2D.')
    blurred = signal.fftconvolve(array, kernel, mode='same')
    return blurred

def convolve_1d(array1: np.ndarray, func: Callable) -> np.ndarray:
    array1_fft = np.fft.fft(array1)
    array2_fft = np.fft.fft(func(array1))
    new_array = np.fft.ifft(array1_fft * array2_fft.conjugate())
    return new_array


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
