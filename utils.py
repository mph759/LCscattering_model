"""
Utilities and other functions
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import time
from pathlib import Path
import inspect
from functools import wraps
import numpy as np


# Specific functions
def generate_positions(space, maximum, change):
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spacial oscillations
    :param space: Spacing in x and y-dimensions between positions
    :param maximum: Maximum values in x and y-dimensions
    :param change: Tuple of allowed deviation from initial lattice spacing
    :return: A position in Cartesian coordinates inside the grid
    """
    # Initial positions, just inside the box
    x_space, y_space = space
    x_change, y_change = change
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


def init_spacing(particle_length, particle_width, unit_vector, padding_spacing):
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
    return x_spacing, y_spacing, displacement


# General functions

def timer(func):
    """
    Function timer
    :param func: Function to be timed
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        result = func(*args, **kwargs)
        t_end = time.perf_counter()
        t_total = t_end - t_start
        func_description = inspect.getdoc(func).split('\n')[0]
        print(f'{func_description} took {t_total:0.4f}s')
        return result

    return wrapper


# Logging
def log_params(params, output_dir='LCscattering_model'):
    Path(f'{output_dir}').mkdir(parents=True, exist_ok=True)
    if not Path(f'{output_dir}/params.log').exists():
        with open(f'{output_dir}/params.log', 'x'):
            pass
    with open(f'{output_dir}/params.log', 'a') as file:
        for arg in params:
            file.write(f'#{arg[0]}\n')
            for param, val in arg[1].items():
                file.write(f'{param.strip("_")}: {val}\n')
            file.write('\n')


# File handling
def fix_file_ext(file_name, file_type):
    """
    Fixes file extension if there is one on the file name which does not match the extension provided
    :param file_name: Potential file name
    :param file_type: Requested file extension
    :return:
    """
    if file_name[-4:] != f".{file_type}":
        file_name = file_name.split('.')[0]
        return f'{file_name}.{file_type}'


def check_existing_ext(file_name):
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


def save(fig, array, file_name, file_type=None, **kwargs):
    """
    Save the figure as a numpy file or as an image
    :param fig: Figure object to be saved
    :param array: numpy array to be saved
    :param file_name: Output file name
    :param file_type: Type of file you want to save (e.g. npy or jpg).
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
    return file_name
