import pandas as pd
from functools import partial
from typing import Optional

from diffraction import Diffraction2D, Diffraction1D
from spatial import load_and_plot_angle_bins, RealSpace
from correlation import AngularCorrelation, plot_2Dcorrelation
from plot_utils import *
from utils import ParameterReader
from compare_1dcorr_exp2sim import simple_postprocessing, postprocessing


def get_folders():
    # Simulated Data
    data_root = Path.cwd() / "output"
    crystal_folder = data_root / r'Crystalline-trial_2026-08-06 14-21-03\unit_vector_60'
    liquid_folder = data_root / r'Liquid-trial_2026-08-19 20-52-02\liquid'
    single_folder = data_root / r'Single-trial_2026-08-07 12-50-32\unit_vector_60'
    nematic_folder = data_root / r'Nematic-trial_2026-08-19 20-01-04\unit_vector_60'
    smectic_folder = data_root / r'LCscattering-trial_2026-09-17 10-41-38\unit_vector_60-padding_spacing_(5, 8)'

    labels = {'Single': single_folder,
              'Crystal': crystal_folder,
              'Smectic': smectic_folder,
              'Nematic': nematic_folder,
              'Liquid': liquid_folder,
              }
    return labels

def plot_model_diffraction(data_folder: Path, label: str, angle_dist: bool = False, save_option: bool = False) -> None:
    if angle_dist:
        fig = plt.figure(figsize=textsize_scale(3 / 5))
        gs0 = GridSpec(nrows=2, ncols=2, figure=fig,
                       width_ratios=[1, 1.1],
                       height_ratios=[1, 0.4],
                       wspace=0.05, hspace=0.3,
                       left=0.11, right=0.9, top=0.95, bottom=0.1)
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
                       left=0.11, right=0.9, top=0.95, bottom=0.05)
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
    if save_option:
        fig.savefig(data_folder / f'LCscattering_{label}.png', dpi=300)
        fig.savefig(data_folder / f'LCscattering_{label}.svg', dpi=300)
        print(f'Saved figure at {data_folder}\LCscattering_{label}')
    plt.close(fig)

def run_all_plot_model_diffraction(save_option: bool= False) -> None:
    for label, folder in get_folders().items():
        if label == 'Smectic' or label == 'Nematic' or label == 'Liquid':
            angle_dist = True
        else:
            angle_dist = False
        plot_model_diffraction(folder, label, angle_dist=angle_dist, save_option=save_option)

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


def plot_2Dcorrelation_comparisons(datasets) -> plt.Figure:
    norm = colors.Normalize(vmin=0, vmax=1e10)
    fig = plt.figure(figsize=textsize_scale(4/5))
    gs0 = GridSpec(3, 3, figure=fig,
                   height_ratios=[1, 1, 0.05],
                   wspace=0.05, hspace=0.3,
                   left=0.075, right=0.95, top=0.95, bottom=0.1)
    ax1 = fig.add_subplot(gs0[0,0])
    ax2 = fig.add_subplot(gs0[0,1], sharey=ax1)
    ax3 = fig.add_subplot(gs0[0,2], sharey=ax1)
    ax4 = fig.add_subplot(gs0[1,0], sharey=ax1)
    ax5 = fig.add_subplot(gs0[1,1], sharey=ax1)
    ax6 = fig.add_subplot(gs0[1,2], sharey=ax1)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    for ax, letter, folder_dir in zip(axes, ascii_lowercase, datasets):
        plot_2Dcorrelation(folder_dir=folder_dir, label=letter, ax=ax)
        if not gs0[:,0]:
            ax.set_ylabel('')
            plt.setp(ax.get_yticklabels(), visible=False)
        if not gs0[1,:]:
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible=False)

    cax = fig.add_subplot(gs0[2,:])
    fig.colorbar(ScalarMappable(norm=norm, cmap=AngularCorrelation.cmap), cax=cax,
                 orientation='horizontal', label=AxesLabel.INTENSITY)
    plt.show()
    return fig

def plot_2Dcorrelation_compare_blanks(datasets):
    fig = plt.figure(figsize=textsize_scale(2/5))
    gs0 = GridSpec(2, 3, figure=fig,
                   height_ratios=[1, 0.05],
                   wspace=0.15, hspace=0.5,
                   left=0.11, right=0.95, top=0.92, bottom=0.15)
    ax1 = fig.add_subplot(gs0[0, 0])
    ax1.set_title('a', loc='left', fontweight='bold')
    cax1 = fig.add_subplot(gs0[1, 0])
    clim1 = 6e4
    norm1 = colors.Normalize(vmin=-clim1, vmax=clim1)

    ax2 = fig.add_subplot(gs0[0, 1], sharex=ax1, sharey=ax1, )
    ax2.set_title('b', loc='left', fontweight='bold')
    ax3 = fig.add_subplot(gs0[0, 2], sharex=ax1, sharey=ax1)
    ax3.set_title('c', loc='left', fontweight='bold')
    cax2 = fig.add_subplot(gs0[1, 1:])
    clim2 = 1.2e10
    norm2 = colors.Normalize(vmin=-clim2, vmax=clim2)

    single = datasets['Single']
    nematic = datasets['Nematic']
    liquid = datasets['Liquid']

    plot_2Dcorrelation(single, ax=ax1, no_cbar=True, norm=norm1)
    fig.colorbar(ScalarMappable(norm=norm1, cmap=AngularCorrelation.cmap), cax=cax1,
                 orientation='horizontal', label=AxesLabel.INTENSITY)
    #cax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plot_2Dcorrelation(nematic, ax=ax2, no_cbar=True, norm=norm2)
    ax2.set_ylabel('')
    plt.setp(ax2.get_yticklabels(), visible=False)

    plot_2Dcorrelation(liquid, ax=ax3, no_cbar=True, norm=norm2)
    ax3.set_ylabel('')
    plt.setp(ax3.get_yticklabels(), visible=False)
    fig.colorbar(ScalarMappable(norm=norm2, cmap=AngularCorrelation.cmap), cax=cax2,
                 orientation='horizontal', label=AxesLabel.INTENSITY)
    #cax2.set_yticks([])
    cax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()
    return fig

def plot_2Dcorrelation_compare_crystal_smectic(datasets):
    fig = plt.figure(figsize=textsize_scale(2/5))
    gs0 = GridSpec(2, 2, figure=fig,
                   height_ratios=[1, 0.05],
                   wspace=0.1, hspace=0.5,
                   left=0.11, right=0.96, top=0.92, bottom=0.15)
    ax1 = fig.add_subplot(gs0[0, 0])
    ax1.set_title('a', loc='left', fontweight='bold')
    cax1 = fig.add_subplot(gs0[1, 0])
    cax1.set_yticks([])
    cax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    clim1 = 2e7
    norm1 = colors.Normalize(vmin=-clim1, vmax=clim1)

    ax2 = fig.add_subplot(gs0[0, 1], sharex=ax1, sharey=ax1)
    ax2.set_title('b', loc='left', fontweight='bold')
    plt.setp(ax2.get_yticklabels(), visible=False)
    #ax2.set_yticks([])

    cax2 = fig.add_subplot(gs0[1, 1])
    cax2.set_yticks([])
    cax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    clim2 = 5e9
    norm2 = colors.Normalize(vmin=-clim2, vmax=clim2)

    crystal = datasets['Crystal']
    smectic = datasets['Smectic']

    plot_2Dcorrelation(crystal, ax=ax1, no_cbar=True, norm=norm1)

    plot_2Dcorrelation(smectic, ax=ax2, no_cbar=True, norm=norm2)
    ax2.set_ylabel('')

    fig.colorbar(ScalarMappable(norm=norm1, cmap=AngularCorrelation.cmap), cax=cax1,
                 orientation='horizontal', label=AxesLabel.INTENSITY)
    fig.colorbar(ScalarMappable(norm=norm2, cmap=AngularCorrelation.cmap), cax=cax2,
                 orientation='horizontal', label=AxesLabel.INTENSITY)

    plt.show()
    return fig

def plot_1Dcorrelation(folder_dir: Path, peak: int, ax: Optional[plt.Axes], **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=textsize_scale(2/5))
    corr_2d = AngularCorrelation.load(folder_dir)
    corr_2d.mean_subtract_by_line()
    corr_2d.plot_line_w_width(point=peak, ax=ax,
                      func=partial(simple_postprocessing, convolve_kwargs={'amplitude':1,'stddev':5}),
                      **kwargs)
    return ax


def plot_1Dcorrelation_compare_crystal_smectic(datasets, index: Optional[int] = None):
    fig, ax = plt.subplots(figsize=textsize_scale(2/5))
    crystal = datasets['Crystal']
    smectic = datasets['Smectic']

    for dataset, color in zip([crystal, smectic], sns.color_palette('colorblind', n_colors=2)):
        reader = ParameterReader(dataset)
        peaks = reader.params['Peak Locations']
        if index is None:
            peak = peaks[0]
        else:
            peak = peaks[index]
        plot_1Dcorrelation(dataset, ax=ax, peak=peak, color=color)
    plt.show()


if __name__ == '__main__':
    data_folders = get_folders()

    #run_all_plot_model_diffraction(save_option=True)
    #read_particle_angles()

    #plot_model_and_angles(data_folders['Smectic'])
    #plot_diffraction_2d_1d(data_folders['Smectic'])
    # plot_2Dcorrelation_compare_blanks(datasets=data_folders)
    #plot_2Dcorrelation_compare_crystal_smectic(datasets=data_folders)

    plot_1Dcorrelation_compare_crystal_smectic(data_folders)
