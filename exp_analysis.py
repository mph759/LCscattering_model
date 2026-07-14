from functools import partial
from typing import Optional, Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import seaborn as sns
from matplotlib import pyplot

from correlation import AngularCorrelation
from utils import alphanum_key, align_ylim, ParameterReader, normalize, halfedgemask_normalize, convolve_gaussian, \
    subtract_mean, half_edge_mask

from plot_settings import *

def analyse_simulation():
    data_root = Path(fr'C:\Users\Michael_X13\OneDrive - RMIT University\Research\LCscattering_model\output')
    data_path = data_root / r'LCscattering-trial_2026-05-14 20-47-05'
    parameter = 'unit_vector'
    folder_list = sorted(data_path.glob(f'{parameter}_*'), key=alphanum_key)
    fig, ax = plot_sim_correlation(folder_list, title=f'{parameter}', step_size=0, func=postprocessing)

    fig.tight_layout()
    plt.show()



@dataclass
class XFM_Experiment:
    ref_num: int
    ref_run: int

    def get_runtag(self, run_num: int) -> str:
        xfm_num = self.ref_num - self.ref_run + run_num
        return f'{xfm_num}_{run_num}'


def plot_exp_correlation(data_path: Path, *, scale: float = 1., step: int = 0, ax: Optional[plt.Axes] = None,
                         label: Optional[str] = None, func:Optional[Callable] = None,
                         **kwargs) -> None:
    if ax is None:
        fig, ax = plt.subplots()
    data = np.load(data_path)
    fname = data_path.stem.split('_')[1]
    if label is None:
        label = fname

    sc, scl = 0.05, 0.05
    rmax = 11.31
    rline = 2.08  # CholPel
    # rline = 3.15
    irline = int(data.shape[0] * rline / rmax)
    w = 2

    dline = np.zeros(data.shape[2])
    tmp = data * 0.
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[1]):
            tmp[i, j, :] = data[i, j, :] - 1 * np.average(data[i, j, :])
    pw = 0
    for i in np.arange(data.shape[0]):
        dline[:] = np.sum(np.sum(tmp[irline - w:irline + w, irline - w:irline + w, :] * (i * i) ** pw, 0), 0)

    if func is not None:
        dline = func(dline)
    # plot a line from q=q'
    ax.plot(np.arange(0, 360, 2), (dline * scale) + step, label=label, **kwargs)
    ax.set_xlim([0, 360])


def plot_all_exp_correlation(well: str, step_size: int = 10):
    tags = ['a', 'b']
    cycles = ['Cycle2', 'Cycle3']
    fig, ax = plt.subplots(ncols=len(tags), nrows=len(cycles), figsize=(16, 9), sharex=True, sharey='row')
    fig.suptitle(f'{well}')

    for cycle, ax_row in zip(cycles, ax):
        data_path = root_path / well / cycle
        ax_row[0].set_ylabel(cycle)
        for tag, axes in zip(tags, ax_row):
            folder_list = sorted(data_path.glob(f'*{tags}_*'), key=alphanum_key)
            for n_step, folder in enumerate(folder_list):
                plot_exp_correlation(folder, step=n_step * step_size, ax=axes)
            axes.set_title(tag)
    # fig.legend()
    fig.tight_layout()
    plt.show()


def plot_sim_correlation(data_folders: list[Path], title: Optional[str] = None, step_size: float=1e12, *,
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
    if convolve_kwargs is not None:
        array = convolve_gaussian(array, **convolve_kwargs)
    array = subtract_mean(array, search_override=half_edge_mask)
    array = normalize(array, search_override=half_edge_mask)
    return array


if __name__ == '__main__':

    # Experimental Data
    Martin_19545 = XFM_Experiment(121019, 424)
    root_path = Path(r'C:\Users\Michael_X13\OneDrive - RMIT University\Beamtime\19545_XFM_Martin\data')
    well = r'CholPel\CholPel_W1'
    # plot_all(well)
    run = 455
    cycle = 'Cycle2'
    type_tag = 'a'

    run_tag = Martin_19545.get_runtag(run)
    exp_data_path = root_path / well / cycle / f'{run_tag}_n49999_{type_tag}_correlation_sum.npy'

    # Simulated Data
    data_root = Path(fr"C:\Users\Michael_X13\OneDrive - RMIT University\Research\LCscattering_model\output")
    data_folder = data_root / r'LCscattering-trial_2024-08-03 17-52-54'

    fixed_parameter = 'unit_vector'

    parameter = 'vector_stddev'
    search_string = f'*{fixed_parameter}'
    print(f'Searching for folders at {data_folder} with {search_string} in the name.')
    folder_list = sorted(data_folder.glob(f'*{fixed_parameter}_7*'), key=alphanum_key)
    print(f'Found {len(folder_list)} folders with {search_string} in the name.')
    len_list = 999
    folder_list = [folder_list[i:i+len_list] for i in range(0, len(folder_list), len_list)]

    postprocessing_w_settings = partial(postprocessing, convolve_kwargs={'amplitude':1,'stddev':3})#{'amplitude':1, 'fwhm_L':0, 'fwhm_G':5})
    postprocessing_wo_settings = partial(postprocessing)
    for folder_sublist in folder_list:
        fig, ax = plt.subplots()
        # fig, ax = plot_saved_angular_corr(folder_sublist, step_size=0, ax=ax, func=postprocessing_w_settings)
        fig, ax = plot_sim_correlation(folder_sublist, step_size=0, ax=ax, func=postprocessing_w_settings)
        # Plot Experimental data

        plot_exp_correlation(exp_data_path, scale=1, ax=ax, label='CholPel', color='k', linestyle='--', func=postprocessing_wo_settings)
        if len(folder_sublist) > 5:
            ncols = len(folder_sublist) // 5 + 1
        else:
            ncols = 1
        ax.legend(ncols=ncols, fontsize='small')
        ax.set_ylim(-1.5, 1.1)
        ax.set_xticks(np.arange(0, 360, step=30))
        ax.set_xlim(0, 180)
        fig.suptitle(f'{run_tag} {type_tag}')

        fig.tight_layout()
    plt.show()

