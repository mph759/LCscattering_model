from itertools import chain
from string import ascii_lowercase

from matplotlib.gridspec import GridSpec
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from diffraction import Diffraction2D, Diffraction1D
from spatial import load_and_plot_angle_bins, RealSpace
from plot_utils import *

def plot_model_diffraction(data_folder: Path, label: str, angle_dist: bool = False, save: bool = False) -> None:
    if angle_dist:
        fig = plt.figure(figsize=textsize_scale(3 / 5))
        gs0 = GridSpec(nrows=2, ncols=2, figure=fig,
                       width_ratios=[1, 1.1],
                       height_ratios=[1, 0.4],
                       wspace=0.05, hspace=0.3,
                       left=0.12, right=0.94, top=0.95, bottom=0.1)
        ax0 = fig.add_subplot(gs0[0, 0])
        ax1 = fig.add_subplot(gs0[0, 1])
        ax2 = fig.add_subplot(gs0[1, :])
        ax2.set_title('c')
        load_and_plot_angle_bins(data_folder, ax=ax2)

    else:
        fig = plt.figure(figsize=textsize_scale())
        gs0 = GridSpec(nrows=1, ncols=2, figure=fig,
                       width_ratios=[1, 1.1],
                       wspace=0.05, hspace=0.01,
                       left=0.1, right=0.94, top=0.95, bottom=0.05)
        ax0 = fig.add_subplot(gs0[0])
        ax1 = fig.add_subplot(gs0[1])

    ax0.set_title('a')
    real_space = RealSpace.load(data_folder)
    real_space.plot(ax=ax0)

    ax1.set_title('b',)
    # cax1 = fig.add_subplot(gs0[2], label='cax1')
    if label == 'Single':
        clim = 1.5e2
    else:
        clim = 1e8
    diffraction_2d = Diffraction2D.load(data_folder)
    diffraction_2d.plot(ax=ax1, clim=clim)  # , cax=cax1)

    plt.show()
    if save:
        fig.savefig(data_folder / f'LCscattering_{label}.png', dpi=300)
        fig.savefig(data_folder / f'LCscattering_{label}.svg', dpi=300)
        print(f'Saved figure at {data_folder}\LCscattering_{label}')
    plt.close(fig)

def run_all_plot_model_diffraction() -> None:
    # Simulated Data
    data_root = Path.cwd() / "output"
    crystal_folder = data_root / r'Crystalline-trial_2026-08-06 14-21-03\unit_vector_60'
    liquid_folder = data_root / r'Liquid-trial_2026-08-19 20-52-02\liquid'
    single_folder = data_root / r'Single-trial_2026-08-07 12-50-32\unit_vector_60'
    nematic_folder = data_root / r'Nematic-trial_2026-08-19 20-01-04\unit_vector_60'
    smectic_folder = data_root / r'LCscattering-trial_2026-08-20 15-06-41\vector_stddev_5-unit_vector_70'

    labels = {'Single': single_folder, 'Nematic': nematic_folder, 'Liquid': liquid_folder, 'Crystal': crystal_folder, 'Smectic': smectic_folder}
    for label, folder in labels.items():
        if label == 'Smectic' or label == 'Nematic' or label == 'Liquid':
            angle_dist = True
        else:
            angle_dist = False
        plot_model_diffraction(folder, label, angle_dist=angle_dist)

def plot_model_and_angles(data_folder: Path) -> None:
    fig = plt.figure(figsize=textsize_scale(4/5))
    ax0 = fig.add_subplot()

    ax0.set_title('a')
    real_space = RealSpace.load(data_folder)
    real_space.plot(ax=ax0)

    ax2 = make_axes_locatable(ax0).append_axes("bottom", size="40%", pad=0.8)
    ax2.set_title('b')
    load_and_plot_angle_bins(data_folder, ax=ax2)
    plt.show()

def plot_diffraction_2d_1d(data_folder: Path) -> None:
    fig, (ax0, ax2) = plt.subplots(figsize=textsize_scale(4/5),
                                   nrows=2,
                                   height_ratios=[1, 0.4],
                                   layout='compressed')

    ax0.set_title('a')
    diffraction2d = Diffraction2D.load(data_folder)
    diffraction2d.plot(ax=ax0, clim=1e8)

    ax2.set_title('b')
    diffraction1d = Diffraction1D(diffraction2d)
    diffraction1d.plot(ax=ax2)
    plt.show()

def plot_model_diffraction_w_1d(data_folder: Path) -> None:
    fig = plt.figure(figsize=textsize_scale(4 / 5))
    gs0 = GridSpec(nrows=2, ncols=2, figure=fig,
                   width_ratios=[1, 1.1],
                   height_ratios=[1, 0.4],
                   wspace=0.1, hspace=0.2,
                   left=0.12, right=0.94, top=0.98, bottom=0.1)
    ax0 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    ax2 = fig.add_subplot(gs0[1, 0])
    ax3 = fig.add_subplot(gs0[1, 1])

    ax0.set_title('a')
    real_space = RealSpace.load(data_folder)
    real_space.plot(ax=ax0)

    ax2.set_title('c')
    load_and_plot_angle_bins(data_folder, ax=ax2)

    ax1.set_title('b')
    diffraction2d = Diffraction2D.load(data_folder)
    diffraction2d.plot(ax=ax1, clim=1e8)

    ax3.set_title('d')
    diffraction1d = Diffraction1D(diffraction2d)
    diffraction1d.plot(ax=ax3)
    ax3.set_yticks([])
    plt.show()

def read_particle_angles():
    folder = Path.cwd() / "output" / r'Liquid-trial_2026-08-19 20-52-02\liquid'
    particle_data = pd.read_csv(folder / 'particle_data.csv', index_col=0)
    init_angles = particle_data['init_angle'] % 180
    angles = particle_data['angle'] % 180
    fig, ax = plt.subplots()
    for angle_list, color in zip([init_angles, angles], ['r','b']):
        counts, bins = np.histogram(angle_list, bins=range(0, 180, 5), density=True)
        ax.hist(angle_list, bins=bins, density=True, color=color, alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    smectic_folder = Path.cwd() / "output" / r'LCscattering-trial_2026-08-20 15-06-41\vector_stddev_5-unit_vector_70'
    #run_all_plot_model_diffraction()
    #read_particle_angles()

    #plot_model_and_angles(smectic_folder)
    #plot_model_diffraction_w_1d(smectic_folder)
    plot_diffraction_2d_1d(smectic_folder)


