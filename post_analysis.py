import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re

from correlation import AngularCorrelation
from utils import ParameterReader


if __name__ == '__main__':

    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['mathtext.default'] = 'regular'

    root_path = Path().cwd() / r'output\LCscattering-trial_2024-08-03 17-52-54'
    parameter = 'padding_spacing'
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    data_folders = sorted(root_path.glob(f'{parameter}_*'), key=alphanum_key)
    fig, ax = plt.subplots()
    step_size = 1e12

    for i, data_folder in enumerate(data_folders):
        data_path = (root_path / data_folder)
        reader = ParameterReader(data_folder)
        angular_correlation = AngularCorrelation.load(data_path)
        peaks = sorted(reader.params['Peak Locations'])
        angular_correlation.plot_line(peaks[0], fig=fig, ax=ax, step=step_size*i, label=data_folder.name)
        plt.legend(loc='upper center')
    plt.show()

