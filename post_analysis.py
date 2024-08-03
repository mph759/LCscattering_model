import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from correlation import AngularCorrelation
from utils import ParameterReader


if __name__ == '__main__':
    root_path = Path().cwd() / r'output\LCscattering-trial_2024-07-26 11-38-08'
    data_folders = ['unit_vector_45', ]
    file_name = 'angular_corr.npy'
    fig, ax = plt.subplots()

    for data_folder in data_folders:
        data_path = root_path / data_folder / file_name
        angular_correlation = AngularCorrelation.load(data_path, num_r=4096, num_th=720)
        with ParameterReader(data_path) as reader:
            # reader.data[]
            angular_correlation.plot_line(1172, ax=ax)
    plt.show()

