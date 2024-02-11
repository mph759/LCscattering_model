"""
Generating a 2D real space array image to impose particles on, and functions associated with that
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
from utils import timer, save
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt


class RealSpace:
    def __init__(self, grid):
        self._grid = grid
        print(f"Generating real space array with dimensions {self._grid}")
        self.img = Image.new('L', self._grid, 0)
        self.__set_array__()
        self.__fig__ = self.__ax__ = None

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

    def plot(self, title=None, inset_size: int = 100):
        """
        Plot the 2D real space image
        :param title: Title text for figure
        :param inset_size: Size of the inset zoomed square in pixels. Default 100
        :return:
        """
        print("Plotting real space figure...")
        self.__fig__, self.__ax__ = plt.subplots()
        self.__ax__.imshow(self.array)
        self.__ax__.invert_yaxis()
        if title is not None:
            self.__ax__.set_title(title)
        self.__ax__.set_xlabel('X')
        self.__ax__.set_ylabel('Y')
        self.__fig__.tight_layout()
        axins = self.__ax__.inset_axes([0.8, 0.8, 0.2, 0.2])
        axins.imshow(self.array)
        x_c, y_c = self.grid[0] * (3 / 4), self.grid[1] * (3 / 4)
        ins_size = inset_size
        x1, x2 = x_c - ins_size / 2, x_c + ins_size / 2
        y1, y2 = y_c - ins_size / 2, y_c + ins_size / 2
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        self.__ax__.indicate_inset_zoom(axins)

    def save(self, file_name, file_type=None, **kwargs):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig__, self.array, file_name, file_type, **kwargs)
        print(f'Saved real space as {file_name}')
