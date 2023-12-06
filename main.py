"""
Generating modelled liquid crystals in 2D
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2023-11-23
"""

import matplotlib.pyplot as plt
import numpy as np
from spatial import RealSpace
from diffraction import DiffractionPattern
from utils import generate_positions, generate_angles, pythagorean_sides
from particle_types import CalamiticParticle

if __name__ == "__main__":
    # Initialise real space parameters
    x_max = y_max = int(1e4)

    # Initialise particle parameters
    particle_width = 2
    particle_length = 15
    # Note: The unit vector is not the exact angle all the particles will have, but the mean of all the angles
    unit_vector = 70  # Unit vector of the particles, starting point up
    vector_stddev = 5  # Standard Deviation of the angle, used to generate angles for individual particles

    # Initialise how the particles sit in real space
    padding_spacing = (5, 5)

    # Initialise beam and detector parameters
    wavelength = 0.67018e-10
    pixel_size = 75e-6
    npt = 2000          # No. of points on the detector

    plt.rcParams['figure.figsize'] = [10, 10]
    ##################### ONLY MODIFY ABOVE #####################
    # Allow spacing in x and y to account for the size and angle of the particle
    x_spacing, y_spacing = (spacing + padding
                            for spacing, padding
                            in zip(pythagorean_sides(particle_length, particle_width, unit_vector), padding_spacing))
    print(f'x spacing: {x_spacing}, y spacing: {y_spacing}')

    # Allow for particles to move slightly in x and y, depending on the spacing
    wobble_allowance = tuple([np.floor((spacing - 1) / 2) for spacing in padding_spacing])

    # Initialise the generators of the positions and angles for the particles
    positions = generate_positions((x_spacing, y_spacing), (x_max, y_max), wobble_allowance)
    angles = generate_angles(unit_vector, vector_stddev)

    # Create the space for the particles
    real_space = RealSpace((x_max, y_max))

    # Generate the particles
    particles = [CalamiticParticle(position, particle_width, particle_length, angle)
                 for position, angle in zip(positions, angles)]
    # Check unit vector matches expected value
    particles_unit_vector = np.mean([particle.angle for particle in particles])
    # print(f"Collective unit vector: {particles_unit_vector:0.2f}")

    # Place particles in real space
    real_space.add(particles)
    # real_space_title = f'Liquid Crystal Phase of Calamitic Liquid crystals, with unit vector {unit_vector}$^\circ$'
    # real_space.plot(real_space_title)

    # Generate diffraction patterns in 2D and 1D of real space
    diffraction_pattern_of_real_space = DiffractionPattern(real_space, wavelength, pixel_size, npt)
    # diffraction_pattern_title = f'2D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
    # diffraction_pattern_of_real_space.plot_2d(diffraction_pattern_title, clim=1e8)
    diff_1D_title = f'1D Diffraction pattern of Liquid Crystal Phase of Calamitic Particles'
    diffraction_pattern_of_real_space.plot_1d(diff_1D_title)

    # Save 1D diffraction pattern as a numpy file
    filename = f'calamitic_p{particle_length}x{particle_width}_uv{unit_vector}'
    diffraction_pattern_of_real_space.save_1d(filename)

    plt.show()

