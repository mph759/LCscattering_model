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

    def plot(self, title=None):
        """
        Plot the 2D real space image
        :param title: Title text for figure
        :param show: Boolean for whether to show the plot immediately. Default False
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

    def save(self, file_name, file_type=None, **kwargs):
        """
        Save the 1D diffraction pattern as a numpy file
        :param file_name: Output file name
        :param file_type: Type of file you want to save (e.g. npy or jpg). Default npy file
        :return:
        """
        file_name = save(self.__fig__, self.array, file_name, file_type, **kwargs)
        print(f'Saved real space as {file_name}')
