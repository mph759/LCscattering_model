"""
Generating 2D scattering pattern for modelled liquid crystals
Original Author: Campbell Tims
Edited by Michael Hassett from 2023-11-23
"""

import numpy as np
import matplotlib.pyplot as plt
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

## Define values ##
xmax = ymax = 1000                      # Size of image along X and Y
grid_size = (xmax, ymax)                # Size of the grid
y_spacing = 20                          # Spacing between lines in y (Vertical)
dy = 1                                  # Variation in y
r = 5                                   # Radius of the scattering point (image)
L = y_spacing * (ymax // y_spacing)     # L = real space length. Units depends on y_spacing
clim = 1e7                              # Diffraction image coolour bar limiter
angle_1, angle_2 = 45, -45              # Angles for ripple pattern
num_repeats_angle_1 = 5                 # Number of times to repeat angle_1
num_repeats_angle_2 = 2                 # Number of times to repeat angle_2
num_points = ((xmax // r) //
              (num_repeats_angle_1 + num_repeats_angle_2))  # Variable for the number of points generated along x


# Create list of points.
def generate_ripple_points(x, y, angle_1, angle_2, r, num_repeats_angle_1, num_repeats_angle_2, point_positions):
    for i in range(num_repeats_angle_1):
        x = x + r * np.cos(np.deg2rad(angle_1))
        y = y + r * np.sin(np.deg2rad(angle_1))
        point_positions.append([x, y])

    for i in range(num_repeats_angle_2):
        x = x + r * np.cos(np.deg2rad(angle_2))
        y = y + r * np.sin(np.deg2rad(angle_2))
        point_positions.append([x, y])


# Create empty list to add points to
point_positions = []

# Variable for the y-position of points generated along y
y_lines = list(range(-ymax, ymax, y_spacing))

# Generate ripple points with different numbers of repeats for angle_1 and angle_2
for i in y_lines:
    generate_ripple_points(0, i, angle_1, angle_2, 2 * r, num_repeats_angle_1, num_repeats_angle_2, point_positions)
    for j in range(num_points):
        generate_ripple_points(point_positions[-1][0], point_positions[-1][1], angle_1, angle_2, 2 * r,
                               num_repeats_angle_1, num_repeats_angle_2, point_positions)

# Vary the point positions
for i, p in enumerate(point_positions):
    point_positions[i][1] += round(np.random.normal(0, dy))

## Placing point coordinates ##
# Create an empty grid for our real space
real_space = np.zeros(grid_size)

# Place points at specified positions
for position in point_positions:
    x, y = position
    iy = int(y)
    if iy >= ymax or iy < 0:
        continue
    ix = int(x)
    if ix >= xmax or ix < 0:
        continue
    real_space[iy, ix] = 1


## Place our image at the point cordinates in real space
# Create the image (simple circle)
def fill_circle(radius, xmax, ymax):
    space = np.zeros((xmax, ymax))

    for x in range(xmax):
        for y in range(ymax):
            dx = x - radius
            dy = y - radius
            if dx ** 2 + dy ** 2 <= radius ** 2:
                space[x, y] = 1
    return space


image = filled_circle = fill_circle(r, xmax, ymax)

# Take the fourier transform of both real_space and the image
Fi = np.fft.fft2(image)
Fr = np.fft.fft2(real_space)

# Convolve the two to place the image at the positions of the scattering points
real_space_c = np.fft.ifft2(Fi.conjugate() * Fr)

# Plot real_space_c
plt.figure(figsize=(8, 8))
plt.imshow(np.abs(real_space_c), extent=[0, L, 0, L])
plt.title("Real Space")
plt.xlabel('X (in units of nm)')
plt.ylabel('Y (in units of nm)')
plt.colorbar()
plt.show()

# Create diffraction image of real_space_c

## Applying the Fourier transform to create a diffraction image
F = np.fft.fft2(real_space_c)
F = np.roll(F, xmax // 2, 0)
F = np.roll(F, ymax // 2, 1)

diffraction_image = np.abs(F)

# Elimintate the centre pixel
diffraction_image[ymax // 2][xmax // 2] = 0

# Plot the diffraction image
plt.figure(figsize=(8, 8))
plt.imshow(diffraction_image ** 2)
plt.title("Diffraction Image")
plt.xlabel(r'X (in units of nm$^{-1}$)')
plt.ylabel(r'Y (in units of nm$^{-1}$)')
plt.colorbar()
plt.clim([0, clim])
plt.show()


## 1D Pattern ##
def frm_integration(frame, unit="q_nm^-1", npt=2250):
    """
    Perform azimuthal integration of frame array
    :param frame: numpy array containing 2D intensity
    :param unit:
    :param npt:
    :return: two-col array of q & intensity.
    """
    # print("Debug - ", self.cam_length, self.pix_size, self.wavelength)
    wavelength = 1e-10
    pix_size = 5e-5
    cam_length = pix_size * L / wavelength
    image_center = np.array(frame.shape) // 2
    ai = AzimuthalIntegrator()
    ai.setFit2D(directDist=cam_length / 1000,
                centerX=image_center[0],
                centerY=image_center[1],
                pixelX=pix_size, pixelY=pix_size)
    ai.wavelength = wavelength
    integrated_profile = ai.integrate1d(data=frame, npt=npt, unit=unit)
    return np.transpose(np.array(integrated_profile))


npt = 2000
diffraction_plot = frm_integration(diffraction_image, unit="q_nm^-1", npt=npt)
radius = xmax // 2  # Assuming you have defined xmax and ymax somewhere
filled_circle = fill_circle(radius, xmax, ymax)

diffraction_image_cone = filled_circle * diffraction_image

diffraction_plot = frm_integration(diffraction_image_cone, unit="q_nm^-1", npt=npt)

non_zero = diffraction_plot[:, 1] != 0  # Removes data points which = 0 due to the cone restriction
diffraction_plot_filtered = diffraction_plot[non_zero]

# Plot 1D integration
plt.figure(figsize=(8, 8))
plt.title("1D Radial Integration")
plt.plot(diffraction_plot_filtered[int(npt // 20):, 0], diffraction_plot_filtered[int(npt // 20):, 1])
plt.xlabel("q_nm$^{-1}$")
plt.ylabel('Arbitary Intensity')
plt.show()
