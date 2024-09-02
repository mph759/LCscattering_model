import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from correlation import AngularCorrelation
from utils import ParameterReader, align_ylim, alphanum_key

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['mathtext.default'] = 'regular'

    root_path = Path().cwd() / r'output\LCscattering-trial_2024-08-03 17-52-54'
    parameter = 'unit_vector'
    data_folders = sorted(root_path.glob(f'{parameter}_*'), key=alphanum_key)
    fig, ax = plt.subplots()
    step_size = 1e12

    for i, data_folder in enumerate(data_folders):
        data_path = (root_path / data_folder)
        reader = ParameterReader(data_folder)
        angular_correlation = AngularCorrelation.load(data_path)
        peaks = sorted(reader.params['Peak Locations'])
        angular_correlation.plot_line(peaks[0], fig=fig, ax=ax, step=step_size * i, label=data_folder.name.split('_')[-1])
    ax.legend(ncol=(i // 5) + 1)
    align_ylim(ax, x_range=(0, angular_correlation.num_th // 2), edge_mask=2)
    plt.title(parameter)
    fig.tight_layout()
    plt.show()
