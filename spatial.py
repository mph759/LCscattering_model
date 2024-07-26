"""
Generating a 2D real space array image to impose particles on, and functions associated with that
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw

from utils import timer, save


class RealSpace:
    def __init__(self, grid):

        self._grid = grid
        print(f"Generating real space array with dimensions {self._grid}")
        self.img = Image.new('L', self._grid, 0)
        self.__set_array__()
        self.__fig__ = self.__ax__ = None
        self.__fig_zoom__ = self.__ax_zoom__ = None

    @property
    def grid(self):
        return self._grid

    @property
    def params(self):
        return ("Space",
                {'grid size': self.grid})

    @timer
    def add(self, particle_list):
        """Adding particles to real space
        :param particle_list: List of particle objects to be added to the real space object
        """
        print(f'No. of Particles: {len(particle_list)}')
        in_real_space = ImageDraw.Draw(self.img)
        for particle in particle_list:
            # print(f'Start: {particle.position}, End: {particle.end_position}')
            particle.create(in_real_space)
        self.__set_array__()

    def __set_array__(self):
        self.array = np.asarray(self.img)

    def plot(self, title=None, inset_size: int = 100, *args, **kwargs):
        """
        Plot the 2D real space image
        :param title: Title text for figure
        :param inset_size: Size of the inset zoomed square in pixels. Default 100
        :return:
        """
        print("Plotting real space figure...")
        self.__fig__, self.__ax__ = plt.subplots()
        self.__ax__.imshow(self.array, *args, **kwargs)
        self.__ax__.invert_yaxis()
        if title is not None:
            self.__ax__.set_title(title)
        self.__ax__.set_xlabel('X')
        self.__ax__.set_ylabel('Y')
        self.__fig__.tight_layout()
        axins = self.__ax__.inset_axes([0.6, 0.6, 0.4, 0.4])
        axins = self.plot_zoom(title=None, zoom_size=inset_size, axes=axins)
        self.__ax__.indicate_inset_zoom(axins)

    def plot_zoom(self, title=None, zoom_size: int = 100, axes=None, *args, **kwargs) -> tuple[Image, Image]:
        """
        Plot the 2D real space image, zoomed in
        :param title: Title text for figure
        :param zoom_size: Size of the zoomed square in pixels. Default 100
        :param axes: Axes object used for plotting. Default None
        :return:
        """
        if axes is None:
            print("Plotting zoomed-in real space figure...")
            self.__fig_zoom__, self.__ax_zoom__ = plt.subplots()
        else:
            self.__ax_zoom__ = axes
        self.__ax_zoom__.imshow(self.array, *args, **kwargs)
        x_c, y_c = self.grid[0] * 0.5, self.grid[1] * 0.5
        x1, x2 = x_c - zoom_size / 2, x_c + zoom_size / 2
        y1, y2 = y_c - zoom_size / 2, y_c + zoom_size / 2
        if title is not None:
            self.__ax_zoom__.set_title(title)
        self.__ax_zoom__.set_xlim(x1, x2)
        self.__ax_zoom__.set_ylim(y1, y2)
        self.__ax_zoom__.set_xticks([])
        self.__ax_zoom__.set_yticks([])
        return self.__ax_zoom__

    def save(self, file_name, file_type=None, **kwargs):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        if self.__fig_zoom__:
            self.__fig_zoom__.savefig(f'{file_name}_zoom.{file_type}', format=file_type, **kwargs)
        file_name = save(self.__fig__, self.array, file_name, file_type, **kwargs)
        print(f'Saved real space as {file_name}')
