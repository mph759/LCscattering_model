import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Callable

from correlation import AngularCorrelation
from utils import ParameterReader, align_ylim, alphanum_key

def plot_saved_angular_corr(data_folders: list[Path], title: str, step_size: float=1e12, *,
                            func: Optional[Callable] = None,
                            peak_num: int = 1, peak_override: int | None = None, legend_nrows:int = 5):
    num_folders = len(data_folders)
    fig, ax = plt.subplots()
    for i, data_folder in enumerate(data_folders):
        angular_correlation = AngularCorrelation.load(data_folder)
        if not peak_override:
            reader = ParameterReader(data_folder)
            peaks = sorted(reader.params['Peak Locations'])
            peak = peaks[peak_num-1]
        else:
            peak = peak_override
        angular_correlation.plot_line(peak, fig=fig, ax=ax, step=step_size * i, label=data_folder.name.split('_')[-1], func=func)
        if i == num_folders - 1:
            align_ylim(ax, x_range=(0, angular_correlation.num_th // 2), edge_mask=2)
    if num_folders > 1:
        ax.legend(ncol=((num_folders-1) // legend_nrows) + 1)
    else:
        ax.legend(ncol=1)
    plt.title(title)
    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['mathtext.default'] = 'regular'

    root_path = Path().cwd() / r'output\LCscattering-trial_2024-09-24 17-38-11'
    parameter = 'unit_vector'
    folder_list = sorted(root_path.glob(f'{parameter}_*'), key=alphanum_key)
    plot_saved_angular_corr(folder_list, title=f'{parameter}', step_size=1e12)
    plt.show()

