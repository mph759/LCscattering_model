"""
Generating modelled liquid crystals in 2D
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import json
import logging
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from correlation import PolarDiffraction2D, AngularCorrelation
from diffraction import Diffraction2D, Diffraction1D
from particle_types import CalamiticParticle
from peak_predict import peak_predict
from spatial import RealSpace
from utils import logger_setup, generate_positions, init_spacing, plot_angle_bins, ParameterLogger


def main():
    """
    Main function. Should not have to modify this
    """
    # Initialise real space parameters
    x_max = y_max = int(2 ** 13)

    # Initialise particle parameters
    particle_width = 2  # in pixels
    particle_length = 15  # in pixels
    # Note: The unit vector is not the exact angle all the particles will have, but the mean of all the angles
    unit_vector = 45
    vector_stddev = 5  # Standard Deviation of the angle, used to generate angles for individual particles

    # Initialise how the particles sit in real space
    padding_spacing = (5, 5)

    # Initialise beam and detector parameters
    wavelength = 0.67018e-10  # metres
    pixel_size = 75e-6  # metres per pixel
    npt = 2000  # No. of points for the radial integration
    dx = 5e-9  # metres

    # Figure formatting - change to suit preference
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['mathtext.default'] = 'regular'

    now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir_root = f'output\\LCscattering-trial_{now}'
    Path(output_dir_root).mkdir(parents=True, exist_ok=True)
    main_logger = logger_setup('main', output_dir_root, stream=True)
    base_run_partial = partial(run, output_dir_root=output_dir_root, grid_max=x_max, particle_width=2,
                               padding_spacing=padding_spacing, wavelength=wavelength, pixel_size=pixel_size,
                               dx=dx, npt=npt)

    # Run over many variables
    variables = {
        "unit_vector": list(range(unit_vector, 90 + vector_stddev, vector_stddev)),
        "vector_stddev": list(range(vector_stddev, 20, vector_stddev)),
        # "particle_width": list(range(particle_width, np.floor_divide(particle_length, particle_width) + 1, 1)),
        "particle_length": list(range(particle_length, 2 * particle_length + 1, 1)),
        "padding_spacing": [(5, x) for x in range(-5, 10 + 1, 1)],
    }
    with open(f'{output_dir_root}/variables.json', 'w') as f:
        json.dump(variables, f)

    start = time.perf_counter()
    kwargs = {}
    for i, (k, v) in enumerate(variables.items()):
        for key, val_range in variables.items():
            if k != key:
                kwargs[key] = val_range[0]
        partial_func = partial(base_run_partial, **kwargs)
        print(f'Resetting partial function with {kwargs}')

        for val in v:
            toc = time.perf_counter()
            print(f'{k} = {val}')
            partial_func(**{k: val, 'tag': f'{k}_{val}'})
            tic = time.perf_counter()
            main_logger.info(f'{k} {val} is done in {tic - toc}s\n')

    end = time.perf_counter()
    run_time = timedelta(seconds=(end - start))
    main_logger.info(f'Total run time: {run_time}\n')


def run(unit_vector, vector_stddev, particle_width, particle_length, *, padding_spacing, output_dir_root, tag, grid_max,
        wavelength, pixel_size, dx, npt):
    output_directory = f'{output_dir_root}\\{tag}'
    with ParameterLogger(output_directory) as log:
        logger = logger_setup('run', output_directory, level=logging.DEBUG)

        # Initialise the spacing in x and y, and the allowed displacement from that lattice
        spacing, allowed_displacement = init_spacing(particle_length, particle_width,
                                                     unit_vector, padding_spacing)
        x_spacing, y_spacing = spacing
        x_spacing = padding_spacing[0] + particle_width

        # Initialise the generators of the positions and angles for the particles
        positions = generate_positions((x_spacing, y_spacing), (grid_max, grid_max), allowed_displacement)
        # Generate the particles
        logger.debug('Generating positions and angles')
        particles = [CalamiticParticle(position, particle_width, particle_length, unit_vector, vector_stddev)
                     for position in positions]
        # Check unit vector matches expected value
        particle_angles = [particle.angle for particle in particles]
        particles_unit_vector = np.mean(particle_angles)
        particles_stddev = np.std(particle_angles)
        fig_angle_dist = plot_angle_bins(particle_angles, unit_vector, vector_stddev)
        fig_angle_dist.savefig(f'{output_directory}\\angle_dist.png')
        logger.info(
            f"Collective unit vector: {particles_unit_vector:0.2f}, with a standard deviation of {particles_stddev:0.2f}")

        # Create the space for the particles and place them in real space
        real_space = RealSpace((grid_max, grid_max))
        real_space.add(particles)

        # Generate diffraction patterns in 2D of real space
        diffraction_of_real_space = Diffraction2D(real_space, wavelength, pixel_size, dx, npt)

        # Log particle information
        particle_params = particles[0].params
        particle_params[1].update({'unit_vector': unit_vector, 'unit_vector_stddev': vector_stddev,
                                   'real_unit_vector': particles_unit_vector,
                                   'real_unit_vector_stddev': particles_stddev,
                                   'no. particles': len(particles)})
        log.params(particle_params)

        # Plot all figures showing, real space, diffraction in 2D and 1D, and the correlation
        # real_space_title = f'Liquid Crystal Phase of Calamitic Liquid crystals, with unit vector {unit_vector}$^\circ$'
        real_space_title = None
        real_space.plot(real_space_title)
        real_space.plot_zoom(real_space_title)
        real_space.save(f'{output_directory}\\2D_model_example', file_type='png', dpi=300, bbox_inches='tight')
        real_space_params = real_space.params
        spatial = {'x_spacing': x_spacing, 'y_spacing': y_spacing,
                   'allowed_random_displacement': allowed_displacement}
        real_space_params[1].update(spatial)
        log.params(real_space_params)

        # diffraction_pattern_title = f'2D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
        diffraction_pattern_title = None
        # Determine the location of peaks
        peak_locs = peak_predict(diffraction_of_real_space, (x_spacing, y_spacing))
        peak_locs = list(set(peak_locs))
        diffraction_of_real_space.plot(diffraction_pattern_title, clim=1e8, peaks=peak_locs)
        # plt.show()
        diffraction_of_real_space.save(f'{output_directory}\\diffraction_pattern_2d', file_type='png',
                                       dpi=300, bbox_inches='tight')
        log.params(diffraction_of_real_space.params)

        # Create and plot 1D diffraction pattern of list of 2D diffraction patterns
        diffraction_pattern_1d = Diffraction1D(diffraction_of_real_space)
        # diff_1D_title = f'1D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
        diff_1D_title = None
        diffraction_pattern_1d.plot(diff_1D_title)
        diffraction_pattern_1d.save(f'{output_directory}\\diffraction_pattern_1d', file_type='png',
                                    dpi=300, bbox_inches='tight')

        log.params({"Peak Locations": {"peaks": peak_locs}})

        # Perform correlation from the diffraction pattern
        polar_plot = PolarDiffraction2D(diffraction_of_real_space,
                                        num_r=diffraction_of_real_space.num_pixels // 2, num_th=720)
        polar_plot.subtract_mean_r()
        polar_plot.gaussian_convolve()

        polar_plot.plot(clim=5e4)
        # plt.show()
        polar_plot.save(f'{output_directory}\\polar_plot', file_type='png', dpi=300, bbox_inches='tight')
        log.params(polar_plot.params)

        angular_corr = AngularCorrelation(polar_plot)
        angular_corr.plot(clim=5e11)
        # plt.show()
        angular_corr.save(f'{output_directory}\\angular_corr', file_type='npy',
                          close_fig=False)
        angular_corr.save(f'{output_directory}\\angular_corr', file_type='png',
                          dpi=300, bbox_inches='tight')

        for peak in peak_locs:
            angular_corr.plot_line(peak,
                                   title=None,
                                   # f'Angular line plot at {q}, with unit vector {unit_vector}',
                                   save_fig=True,
                                   save_name=f'{output_directory}\\angular_line_{peak}',
                                   save_type='png', dpi=300, bbox_inches='tight')

        # plt.show()
        plt.close('all')


if __name__ == "__main__":
    main()
