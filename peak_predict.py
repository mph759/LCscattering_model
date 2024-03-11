"""
Peak predict in an FFT from real space parameters
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2024-03-10
"""

import numpy as np
from diffraction import Diffraction1D


def peak_predict(diffraction: Diffraction1D, num_pixels: tuple[int], d_spacings: tuple[float]):
    """
    Predict the peak postion (q) from real space parameters
    :return: peak_locs: peak positions in q
    """
    peak_locs = []
    peak_locs_theor = sorted(np.round(np.divide(num_pixels, d_spacings)))
    for peak in peak_locs_theor:
        print(f"peak: {peak}")
        masked_pixels = np.ma.masked_outside(diffraction.pattern_1d[:, 1],
                                             diffraction.pattern_1d[int(peak - 2), 1],
                                             diffraction.pattern_1d[int(peak + 2), 1])
        peak_index = np.argmax(masked_pixels)
        peak_locs.append((peak_index, diffraction.pattern_1d[peak_index, 0]))
        print(f"peak index: {peak_index}")
        print(f"peak val: {diffraction.pattern_1d[peak_index, 0]}")
    return peak_locs
