"""
Generating modelled liquid crystals in 2D
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from correlation import PolarAngularCorrelation
from diffraction import DiffractionPattern
from spatial import RealSpace
from utils import generate_positions, init_spacing, log_params
from particle_types import CalamiticParticle

from multiprocessing import Pool
from functools import partial


def main():
    """
    Main function. Should not have to modify this
    """
    # Initialise real space parameters
    x_max = y_max = int(2**14)

    # Initialise particle parameters
    particle_width = 2  # in pixels
    particle_length = 15  # in pixels
    # Note: The unit vector is not the exact angle all the particles will have, but the mean of all the angles
    unit_vector_init = 45
    unit_vector_fin = 90  # Unit vector of the particles, starting point up
    vector_stddev = 5  # Standard Deviation of the angle, used to generate angles for individual particles

    # Initialise how the particles sit in real space
    padding_spacing = (5, 5)

    # Initialise beam and detector parameters
    wavelength = 0.67018e-10  # metres
    pixel_size = 75e-6  # metres
    npt = 2000  # No. of points for the radial integration
    dx = 5e-9  # metres

    # Figure formatting - change to suit preference
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['mathtext.default'] = 'regular'

    now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir_root = f'output\\LCscattering-{now}'

    '''
    for unit_vector in range(unit_vector_init, unit_vector_fin+vector_stddev, vector_stddev):
        output_dir = f'{output_dir_root}\\unit_vector_{int(unit_vector)}'
        main(output_dir)
        print()
    '''
    unit_vector = range(unit_vector_init, unit_vector_fin + vector_stddev, vector_stddev)
    run_partial = partial(run, output_dir_root=output_dir_root, particle_length=particle_length,
                          particle_width=particle_width, padding_spacing=padding_spacing, x_max=x_max, y_max=y_max,
                          vector_stddev=vector_stddev, wavelength=wavelength, pixel_size=pixel_size, dx=dx, npt=npt)

    start = time.perf_counter()
    with Pool(processes=2) as pool:
        results = pool.imap(run_partial, unit_vector)

        for i, _ in enumerate(results):
            print(f'{unit_vector[i]} Done')

    end = time.perf_counter()
    print(f'Total time: {end - start}')


def run(unit_vector, *, output_dir_root, particle_length, particle_width, padding_spacing, x_max, y_max, vector_stddev,
        wavelength, pixel_size, dx, npt):
    output_directory = f'{output_dir_root}\\unit_vector_{int(unit_vector)}'
    # Initialise the spacing in x and y, and the allowed displacement from that lattice
    x_spacing, y_spacing, allowed_displacement = init_spacing(particle_length, particle_width,
                                                              unit_vector, padding_spacing)
    x_spacing = padding_spacing[0] + particle_width
    spatial = {'x_spacing': x_spacing, 'y_spacing': y_spacing, 'allowed_random_displacement': allowed_displacement}
    # Initialise the generators of the positions and angles for the particles
    positions = generate_positions((x_spacing, y_spacing), (x_max, y_max), allowed_displacement)

    # Generate the particles
    particles = [CalamiticParticle(position, particle_width, particle_length, unit_vector, vector_stddev)
                 for position in positions]
    # Check unit vector matches expected value
    particles_unit_vector = np.mean([particle.angle for particle in particles])
    print(f"Collective unit vector: {particles_unit_vector:0.2f}")

    # Create the space for the particles and place them in real space
    real_space = RealSpace((x_max, y_max))
    real_space.add(particles)

    # Generate diffraction patterns in 2D and 1D of real space
    diffraction_pattern_of_real_space = DiffractionPattern(real_space, wavelength, pixel_size, dx, npt)

    # Perform correlation from the diffraction pattern
    polar_plot = PolarAngularCorrelation(diffraction_pattern_of_real_space,
                                         300, 720, subtract_mean=True)

    # Plot all figures showing, real space, diffraction in 2D and 1D, and the correlation
    # real_space_title = f'Liquid Crystal Phase of Calamitic Liquid crystals, with unit vector {unit_vector}$^\circ$'
    # real_space.plot(real_space_title)
    diffraction_pattern_title = f'2D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
    diffraction_pattern_of_real_space.plot_2d(diffraction_pattern_title, clim=1e8)
    # diff_1D_title = f'1D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
    # diffraction_pattern_of_real_space.plot_1d(diff_1D_title)

    polar_plot.plot(clim=1e4)
    polar_plot.angular_correlation()
    polar_plot.plot_angular_correlation(clim=1e10)

    # Log parameters
    particle_params = particles[0].params
    particle_params[1].update({'unit_vector': unit_vector, 'real_unit_vector': particles_unit_vector})
    real_space_params = real_space.params
    real_space_params[1].update(spatial)
    params = (particle_params, real_space_params,
              diffraction_pattern_of_real_space.params, polar_plot.params)
    log_params(params, output_directory)

    # Save figures as images
    real_space.save(f'{output_directory}\\2D_model', file_type='jpeg', dpi=2000, bbox_inches='tight')
    diffraction_pattern_of_real_space.save_1d(f'{output_directory}\\diffraction_pattern_1d', file_type='jpeg',
                                              dpi=300, bbox_inches='tight')
    diffraction_pattern_of_real_space.save_2d(f'{output_directory}\\diffraction_pattern_2d', file_type='jpeg',
                                              dpi=300, bbox_inches='tight')
    polar_plot.save(f'{output_directory}\\polar_plot', file_type='jpeg', dpi=300, bbox_inches='tight')
    polar_plot.save_angular_correlation(f'{output_directory}\\angular_corr', file_type='jpeg',
                                        dpi=300, bbox_inches='tight')
    radial_lines = [50]
    for line in radial_lines:
        polar_plot.plot_angular_correlation_point(line,
                                                  title=f'Angular line plot at {line}, with unit vector {unit_vector}',
                                                  y_lim=(-2e11, 2e11), save_fig=True,
                                                  save_name=f'{output_directory}\\angular_line_{line}',
                                                  save_type='jpeg', dpi=300, bbox_inches='tight')

    plt.close('all')
    # plt.show()


if __name__ == "__main__":
    main()
