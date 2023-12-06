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
