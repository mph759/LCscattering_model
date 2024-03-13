"""
Peak predict in an FFT from real space parameters
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2024-03-10
"""

import numpy as np
from diffraction import Diffraction2D


def peak_predict(diffraction: Diffraction2D, num_pixels: tuple[int], d_spacings: tuple[float]):
    """
    Predict the peak postion (q) from real space parameters
    :return: peak_locs: peak positions in q
    """
    peak_locs = []
    peak_locs_theor = sorted(np.round(np.divide(num_pixels, d_spacings)))
    centre = num_pixels[0]//2
    for peak in peak_locs_theor:
        for new_peak in range(int(peak), centre, int(peak)):
            if new_peak > centre or new_peak in peak_locs_theor:
                continue
            else:
                peak_locs_theor.append(new_peak)

    diffraction_pattern_half = diffraction.pattern_2d[centre:, centre]

    for i, peak in enumerate(peak_locs_theor):
        min_value = int(peak - 2)
        while min_value < 0:
            min_value += 1
        max_value = int(peak + 2)
        while max_value >= centre:
            max_value -= 1
        masked_pixels = np.ma.masked_outside(diffraction_pattern_half,
                                             diffraction_pattern_half[min_value],
                                             diffraction_pattern_half[max_value])
        peak_index = np.argmax(masked_pixels)
        peak_locs.append(peak_index)
    return peak_locs
