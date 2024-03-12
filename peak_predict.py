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
    diffraction_pattern_half = diffraction.pattern_2d[num_pixels[0] // 2:, num_pixels[0] // 2]
    for peak in peak_locs_theor:
        print(f"predicted peak: {int(peak)}")
        masked_pixels = np.ma.masked_outside(diffraction_pattern_half,
                                             diffraction_pattern_half[int(peak - 2)],
                                             diffraction_pattern_half[int(peak + 2)])
        peak_index = np.argmax(masked_pixels)
        print(f"peak index: {peak_index}")
        peak_locs.append(peak_index)
    return peak_locs
