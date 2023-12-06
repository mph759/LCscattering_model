"""
Utilities and other functions
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import time
import inspect
from functools import wraps
import numpy as np


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


def generate_positions(space, max, change):
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spacial oscillations
    :param x_space: Spacing in x-dimension between positions
    :param y_space: Spacing in y-dimension between positions
    :param change: Tuple of allowed deviation from initial lattice spacing
    :return: A position in Cartesian coordinates inside the grid
    """
    # Initial positions, just inside the box
    x_space, y_space = space
    x_change, y_change = change
    x_max, y_max = max
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


def generate_angles(mean_angle: int, angle_stddev: int):
    """
    Generate an angle from a normal distribution, with a given mean and standard deviation
    :param mean_angle: Mean angle of the normal distribution
    :param angle_stddev: standard deviation of the normal distribution
    :return:
    """
    while True:
        angle = np.random.normal(mean_angle, angle_stddev) % 360
        yield angle


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
    if file_type == "npy":
        file_name = fix_file_ext(file_name, file_type)
        np.save(file_name, array)
    else:
        file_name = fix_file_ext(file_name, file_type)
        fig.savefig(file_name, format=file_type, **kwargs)
    return file_name
