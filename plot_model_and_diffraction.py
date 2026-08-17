import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from diffraction import Diffraction2D, RealSpace
from utils import ParameterReader
from plot_settings import *

def plot_model_and_diffraction(data_folder: Path, label: str) -> None:
    reader = ParameterReader(data_folder)
    grid_size = reader.params['Space']['grid size']
    real_space = RealSpace(grid=grid_size, file_name=data_folder / '2D_model.npy')
    fig = plt.figure(figsize=(6.69, 9.33 * (2 / 5)))
    gs0 = GridSpec(nrows=1, ncols=2, figure=fig,
                   width_ratios=[1, 1.1,],
                   wspace=0.05, hspace=0.01,
                   left=0.1, right=0.94, top=0.95, bottom=0.05)

    ax0 = fig.add_subplot(gs0[0])
    ax0.set_title('a', loc='left', fontweight='bold')

    real_space.plot(ax=ax0)

    wavelength = reader.params['Diffraction']['wavelength']
    pixel_size = reader.params['Diffraction']['pixel_size']
    dx = reader.params['Diffraction']['dx']
    npt = reader.params['Diffraction']['npt']
    diffraction_2d = Diffraction2D(real_space, wavelength=wavelength, dx=dx, npt=npt, pixel_size=pixel_size)

    ax1 = fig.add_subplot(gs0[1])
    ax1.set_title('b', loc='left', fontweight='bold')
    #cax1 = fig.add_subplot(gs0[2], label='cax1')
    if label == 'Single':
        clim = 1.5e2
    else:
        clim = 1e8
    diffraction_2d.plot(ax=ax1, clim=clim) #, cax=cax1)
    plt.show()
    fig.savefig(data_folder / f'LCscattering_{label}.png', dpi=300)
    fig.savefig(data_folder / f'LCscattering_{label}.svg', dpi=300)
    print(f'Saved figure at {data_folder}\LCscattering_{label}')
    # plt.show()

def plot_model_diffraction_and_angle_bins(data_folder: Path, label: str) -> None:
    reader = ParameterReader(data_folder)
    grid_size = reader.params['Space']['grid size']
    real_space = RealSpace(grid=grid_size, file_name=data_folder / '2D_model.npy')
    fig = plt.figure(figsize=(6.69, 9.33 * (2 / 5)))
    gs0 = GridSpec(nrows=2, ncols=2, figure=fig,
                   width_ratios=[1, 1.1],
                   height_ratios=[1, 0.6],
                   wspace=0.05, hspace=0.01,
                   left=0.1, right=0.94, top=0.95, bottom=0.05)

    ax0 = fig.add_subplot(gs0[0,0])
    ax0.set_title('a', loc='left', fontweight='bold')

    real_space.plot(ax=ax0)

    wavelength = reader.params['Diffraction']['wavelength']
    pixel_size = reader.params['Diffraction']['pixel_size']
    dx = reader.params['Diffraction']['dx']
    npt = reader.params['Diffraction']['npt']
    diffraction_2d = Diffraction2D(real_space, wavelength=wavelength, dx=dx, npt=npt, pixel_size=pixel_size)

    ax1 = fig.add_subplot(gs0[1,0])
    ax1.set_title('b', loc='left', fontweight='bold')
    # cax1 = fig.add_subplot(gs0[2], label='cax1')
    if label == 'Single':
        clim = 1.5e2
    else:
        clim = 1e8
    diffraction_2d.plot(ax=ax1, clim=clim)  # , cax=cax1)

    ax2 = fig.add_subplot(gs0[:,1])
    ax2.set_title('c', loc='left', fontweight='bold')
    angle_mean = reader.params['Diffra']['angle_mean']

    plt.show()
    fig.savefig(data_folder / f'LCscattering_{label}.png', dpi=300)
    fig.savefig(data_folder / f'LCscattering_{label}.svg', dpi=300)
    print(f'Saved figure at {data_folder}\LCscattering_{label}')
    # plt.show()

if __name__ == '__main__':
    # Simulated Data
    data_root = Path.cwd() / "output"
    crystal_folder = data_root / r'Crystalline-trial_2026-08-06 14-21-03\unit_vector_60'
    liquid_folder = data_root / r'Liquid-trial_2026-08-07 10-01-43\liquid'
    single_folder = data_root / r'Single-trial_2026-08-07 12-50-32\unit_vector_60'
    nematic_folder = data_root / r'Nematic-trial_2026-08-07 12-55-08\unit_vector_60'

    labels = {'Single': single_folder, 'Nematic': nematic_folder, 'Liquid': liquid_folder, 'Crystal': crystal_folder}
    for label, folder in labels.items():
        plot_model_and_diffraction(folder, label)

