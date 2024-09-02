import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from correlation import AngularCorrelation
from utils import ParameterReader, align_ylim, alphanum_key

def plot_saved_angular_corr(data_folders: list[Path], title: str, step_size: float=1e12, *,
                            peak_num: int = 0, legend_nrows:int = 5):
    num_folders = len(data_folders)
    fig, ax = plt.subplots()
    for i, data_folder in enumerate(data_folders):
        reader = ParameterReader(data_folder)
        angular_correlation = AngularCorrelation.load(data_folder)
        peaks = sorted(reader.params['Peak Locations'])
        angular_correlation.plot_line(peaks[peak_num], fig=fig, ax=ax, step=step_size * i, label=data_folder.name.split('_')[-1])
    ax.legend(ncol=((num_folders-1) // legend_nrows) + 1)
    align_ylim(ax, x_range=(0, angular_correlation.num_th // 2), edge_mask=2)
    plt.title(title)
    fig.tight_layout()


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['mathtext.default'] = 'regular'

    root_path = Path().cwd() / r'output\LCscattering-trial_2024-08-03 17-52-54'
    parameter = 'unit_vector'
    folder_list = sorted(root_path.glob(f'{parameter}_*'), key=alphanum_key)
    plot_saved_angular_corr(folder_list, title=f'{parameter}', step_size=1e12)
    plt.show()


