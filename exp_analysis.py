import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

from utils import alphanum_key, align_ylim, ParameterReader
from correlation import AngularCorrelation
from post_analysis import plot_saved_angular_corr


@dataclass
class XFM_Experiment:
    ref_num:int
    ref_run:int

    def get_runtag(self, run_num:int) -> str:
        xfm_num = self.ref_num - self.ref_run + run_num
        return f'{xfm_num}_{run_num}'

def display_correlation(data_path: Path, *, scale: float=1, step: int = 0,  ax: plt.Axes | None = None, label: str | None = None,
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


    # plot a line from q=q'
    ax.plot(np.arange(0, 360, 2), (dline*scale)+step, label=label, **kwargs)
    ax.set_xlim([0, 180])

def plot_all(well: str, step_size: int = 10):
    tags = ['a', 'b']
    cycles = ['Cycle2', 'Cycle3']
    fig, ax = plt.subplots(ncols=len(tags), nrows=len(cycles), figsize=(16, 9), sharex=True, sharey='row')
    fig.suptitle(f'{well}')

    for cycle, ax_row in zip(cycles, ax):
        data_path = root_path / well / cycle
        ax_row[0].set_ylabel(cycle)
        for tag, axes in zip(tags, ax_row):
            folder_list = sorted(data_path.glob(f'*{tags}_correlation_sum.npy'), key=alphanum_key)
            for n_step, folder in enumerate(folder_list):
                display_correlation(folder, step=n_step * step_size, ax=axes)
            axes.set_title(tag)
    # fig.legend()
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['mathtext.default'] = 'regular'


    # Experimental Data
    Martin_19545 = XFM_Experiment(121019, 424)
    root_path = Path(r'C:\Users\Michael_X13\OneDrive - RMIT University\Beamtime\19545_XFM_Martin\data')
    well = 'CholPel_W1'
    # plot_all(well)
    run = 528
    cycle = 'Cycle2'
    type_tag = 'a'

    run_tag = Martin_19545.get_runtag(run)
    exp_data_path = root_path / well / cycle / f'{run_tag}_n49999_{type_tag}_correlation_sum.npy'



    # Simulated Data
    data_folder = Path().cwd() / r'output\LCscattering-trial_2024-09-25 13-38-33'

    '''
    fig, ax = plt.subplots(figsize=(16, 9))
    data_folder = data_folder / r'unit_vector_64'
    reader = ParameterReader(data_folder)
    peaks = sorted(reader.params['Peak Locations'])
    unit_vector = reader.params['Calamitic Particle']['unit_vector']
    vector_stddev = reader.params['Calamitic Particle']['unit_vector_stddev']
    angular_correlation = AngularCorrelation.load(data_folder)
    angular_correlation.plot_line(peaks[0], fig=fig, ax=ax,
                                  label='Simulation')
    '''
    parameter = 'unit_vector'
    folder_list = sorted(data_folder.glob(f'{parameter}_*'), key=alphanum_key)
    fig, ax = plot_saved_angular_corr(folder_list, title=f'{parameter}', step_size=0, peak_override=433)

    # Plot Experimental data
    display_correlation(exp_data_path, scale=1e11, ax=ax, label='data', color='k')
    ax.legend()
    fig.suptitle(f'{run_tag} {type_tag}')
    align_ylim(ax=ax, x_range=(0, 180), edge_mask=2)
    fig.tight_layout()
    plt.show()

