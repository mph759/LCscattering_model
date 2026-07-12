import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Callable
import seaborn as sns

from correlation import AngularCorrelation
from utils import ParameterReader, align_ylim, alphanum_key, gaussian_convolve, subtract_mean, normalize, convolve_voigt
from plot_settings import *

def plot_saved_angular_corr(data_folders: list[Path], title: Optional[str] = None, step_size: float=1e12, *,
                            peak_num: int = 1, peak_override: int | None = None, legend_nrows: int = 5,
                            func: Optional[Callable] = None, ax: Optional[plt.Axes] = None,
                            **func_kwargs: dict[str, float]):
    num_folders = len(data_folders)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    for i, (data_folder, color) in enumerate(zip(data_folders, sns.color_palette('colorblind'))):
        angular_correlation = AngularCorrelation.load(data_folder)
        if not peak_override:
            reader = ParameterReader(data_folder)
            peaks = sorted(reader.params['Peak Locations'])
            peak = peaks[peak_num-1]
        else:
            peak = peak_override
        # angular_correlation.ang_corr=gaussian_convolve(angular_correlation.ang_corr, 10, 3)
        angular_correlation.plot_line(peak, fig=fig, ax=ax, step=step_size * i, label=data_folder.name, func=func, color=color, **func_kwargs)
        if i == num_folders - 1:
            align_ylim(ax, x_range=(0, angular_correlation.num_th // 2), edge_mask=2)
    if num_folders > 1:
        ax.legend(ncol=((num_folders-1) // legend_nrows) + 1)
    else:
        ax.legend(ncol=1)
    if title:
        plt.title(title)
    return fig, ax

def postprocessing(array: np.ndarray, convolve_kwargs:Optional[dict] = None) -> np.ndarray:
    # Perform convolvution and mean subtraction on array
    if convolve_kwargs is None:
        convolve_kwargs = {'amplitude':1,
                           'fwhm_L':1,
                           'fwhm_G':10,}
    array = convolve_voigt(array, **convolve_kwargs)
    array = subtract_mean(array)
    array = normalize(array)
    return array

if __name__ == '__main__':
    data_root = Path(fr'C:\Users\Michael_X13\OneDrive - RMIT University\Research\LCscattering_model\output')
    data_path = data_root / r'LCscattering-trial_2026-05-14 20-47-05'
    parameter = 'unit_vector'
    folder_list = sorted(data_path.glob(f'{parameter}_*'), key=alphanum_key)
    fig, ax = plot_saved_angular_corr(folder_list, title=f'{parameter}', step_size=0, func=postprocessing)

    fig.tight_layout()
    plt.show()

