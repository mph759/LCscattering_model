"""
Generating modelled liquid crystals in 2D
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
from pathlib import Path
import pandas as pd
from typing import Generator, Any, TypeAlias, Callable, Optional

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from plot_settings import *
from utils import ParameterReader

Coordinates2D: TypeAlias = tuple[int, int]


class PointParticle:
    def __init__(self, position: tuple[int, int]):
        """
        A point particle in real space
        :param position: Position of the particle in Cartesian coordinates
        """
        self._position = position  # Position of the particle in real space using cartesian coordinates

    @property
    def position(self) -> Coordinates2D:
        return self._position

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def params(self):
        return {'Point Particle', {}}

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The point to be drawn on the real space object
        """
        return draw_object.point(self.position, fill=1)

    @property
    def position_data(self):
        return [self.x, self.y]


class CalamiticParticle(PointParticle):
    def __init__(self, init_position: tuple[int, int], width: int, length: int, angle_func: Callable):
        """
        A calamitic (rod-like) particle in real space
        :param init_position: Position of the particle in Cartesian coordinates
        :param width: Width of the particle
        :param length: Length of the particle
        :param mean_angle: Angle of the particle in real space
        """
        super().__init__(init_position)
        self._width = width
        self._length = length
        self._get_end_points(angle_func)

    @property
    def width(self):
        return self._width

    @property
    def length(self):
        return self._length

    @property
    def angle(self):
        return self._angle

    @property
    def init_angle(self):
        return self._init_angle

    @property
    def end_position(self) -> Coordinates2D:
        return self._end_position

    @property
    def x1(self):
        return self.position[0]

    @property
    def y1(self):
        return self.position[1]

    @property
    def x2(self):
        return self._end_position[0]

    @property
    def y2(self):
        return self._end_position[1]

    @property
    def params(self):
        return {'Calamitic Particle':
                    {'width': self.width,
                     'length': self.length}}
    @property
    def position_data(self):
        return [self.x1, self.y1, self.x2, self.y2, self.init_angle, self.angle]

    def _get_end_points(self, angle_func: Callable):
        """
        Calculate the coordinates of the end of the particle, given its length and angle
        :return: The end coordinates of the particle
        """
        self._init_angle = angle_func() % 360

        x2 = self.x1 + np.round(self.length * np.cos(np.radians(self._init_angle)))
        y2 = self.y1 + np.round(self.length * np.sin(np.radians(self._init_angle)))
        self._end_position = x2, y2
        if self.x1 == self.x2:
            if self.y1 < self.y2:
                self._angle = 90
            else:
                self._angle = 270
        else:
            self._angle = np.rad2deg(np.arctan((self.y2 - self.y1) / (self.x2 - self.x1))) % 360

    def create(self, draw_object):
        """
        Draw the particle onto real space
        :param draw_object: The RealSpace draw object
        :return: The line to be drawn on the real space object
        """
        return draw_object.line([self.position, self.end_position], fill=1, width=self.width)

def write_particle_data(folder: Path, particle_list: list[CalamiticParticle]):
    file = folder / 'particle_data.csv'
    pd.DataFrame([particle.position_data for particle in particle_list],
                 columns=['x1','y1','x2','y2','init_angle','angle']).to_csv(file)

def init_spacing(particle_length: int, particle_width: int,
                 unit_vector: int, padding_spacing: Coordinates2D) -> tuple[Coordinates2D, Coordinates2D]:
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


def generate_positions(space: Coordinates2D, maximum: Coordinates2D,
                       change: Coordinates2D) -> Generator[Coordinates2D, Any, None]:
    """
    Generate a position inside cartesian coordinates, given a rough lattice with random spatial oscillations
    :rtype: Generator[
    Coordinates2D, Any, None]
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

def generate_random_positions(num_particles: int, maximum: Coordinates2D) -> Generator[Coordinates2D, Any, None]:
    for _ in range(num_particles):
        yield np.random.randint(0, maximum[0]), np.random.randint(0, maximum[1])


def normal_distribution(mean, stddev):
    return np.random.normal(mean, stddev)

def uniform_distribution(min, max):
    return np.random.uniform(min, max)


def exact(value):
    return value


def pythagorean_sides(a: float | int, b: float | int, theta: float | int) -> tuple[int, int]:
    """
    Calculates the side lengths of a right angle triangle using the Pythagorean formulae
    :param a: Length of triangle (a)
    :param b: Width of the triangle (b)
    :param theta: Angle of the triangle
    :return: x and y coordinates of the end point
    """
    theta_radians = np.deg2rad(theta)
    x = abs(np.round(a * np.cos(theta_radians))) + abs(np.round(b * np.sin(theta_radians)))
    y = abs(np.round(a * np.sin(theta_radians))) + abs(np.round(b * np.cos(theta_radians)))
    return x, y

def plot_angle_bins(samples: pd.DataFrame | list, mean: float, stddev: float,
                    ax: Optional[plt.Axes] = None, svg_friendly: bool = False, x_max: float | int = 180) -> tuple[plt.Figure, plt.Axes]:
    sample_mean = np.mean(samples)
    sample_stddev = np.std(samples)
    sample_size = len(samples)
    bins = range(0, 360, 5)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    counts, bins = np.histogram(samples, bins=bins, density=True)
    ax.hist(samples, bins=bins, density=True)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    if stddev != 0:
        y = 1 / (stddev * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * stddev ** 2))
        for i, angle in enumerate(y):
            if angle < 0:
                angle += 360
            angle %= 360
            y[i] = angle
        ax.plot(bins, y, 'r', alpha=0.5)
    ax.xaxis.set_major_locator(mtick.MultipleLocator(30))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(10))
    ax.set_xlim(0, x_max)

    ax.set_ylabel('Frequency')
    if svg_friendly:
        ax.set_xlabel(AxesLabel.ANGLE_SVG)
    else:
        ax.set_xlabel(AxesLabel.ANGLE)
        fig.tight_layout()
    return fig, ax


def plot_angle_bins_polar(samples, mean: float, stddev: float):
    sample_size = len(samples)
    bins = range(0, 360, 1)
    fig = plt.figure(figsize=textsize_square())

    ax = fig.add_subplot(projection='polar')
    counts, bins = np.histogram(samples, bins=bins, density=True)
    area = counts / sample_size
    radius = (area / np.pi) ** (1 / 2)
    ax.bar(np.radians(bins[:-1]), radius, width=1)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xticks(np.radians(range(bins[0], bins[-1], 15)))
    fig.tight_layout()
    return fig, ax

def load_and_plot_angle_bins(folder: Path, **kwargs) -> plt.Axes:
    reader = ParameterReader(folder)
    angle_mean = reader.params['Calamitic Particle']['unit_vector']
    angle_stddev = reader.params['Calamitic Particle']['unit_vector_stddev']
    particle_data = pd.read_csv(folder / 'particle_data.csv', index_col=0)
    angles = particle_data['angle']
    _, ax = plot_angle_bins(angles, angle_mean, angle_stddev, svg_friendly=True, **kwargs)
    return ax

if __name__ == '__main__':
    mean_angle = 60
    angle_stddev = 3
    angles = np.random.normal(mean_angle, angle_stddev, int(1e6))
    for i, angle in enumerate(angles):
        if angle < 0:
            angle += 360
        angle %= 360
        angles[i] = angle
    fig, ax = plot_angle_bins(angles, mean_angle, angle_stddev)
    fig, ax = plot_angle_bins_polar(angles, mean_angle, angle_stddev)
    plt.show()
