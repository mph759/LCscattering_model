"""
Generating a 2D real space array image to impose particles on, and functions associated with that
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
from utils import timer
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt


class RealSpace:
    def __init__(self, grid):
        self.grid = grid
        print(f"Generating real space array with dimensions {self.grid}")
        self.img = Image.new('L', self.grid, 0)
        self.__set_array__()

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

    def plot(self, title, **kwargs):
        """
        Plot the 2D real space image
        :param title: Title text for figure
        :return:
        """
        print("Plotting real space figure...")
        plt.figure()
        plt.imshow(self.array, extent=(0, self.grid[0], 0, self.grid[1]))
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        if 'show' in kwargs and kwargs['show']:
            plt.show()
