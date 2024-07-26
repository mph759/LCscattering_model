"""
Peak predict in an FFT from real space parameters
Project: Generating 2D scattering pattern for modelled liquid crystals
Authored by Michael Hassett from 2024-03-10
"""

import numpy as np

from diffraction import Diffraction2D


class MillerIndex_3D:
    def __init__(self, x, y, z) -> None:
        self._h = x
        self._k = y
        self._l = z

    @property
    def h(self) -> int:
        return self._h

    @property
    def k(self) -> int:
        return self._k

    @property
    def l(self) -> int:
        return self._l

    def __len__(self) -> float:
        return np.sqrt(self.h ** 2 + self.k ** 2 + self.l ** 2)

    def __str__(self) -> str:
        return f"[{self.h}{self.k}{self.l}]"

    def __add__(self, other) -> 'MillerIndex_3D':
        if isinstance(other, MillerIndex_3D):
            return MillerIndex_3D(self.h + other.h, self.k + other.k, self.l + other.l)
        elif isinstance(other, tuple) and len(other) == 3:
            return MillerIndex_3D(self.h + other[0], self.k + other[1], self.l + other[2])
        else:
            raise TypeError(f"{other} is not an appropriate Miller Index (3D)")


class MillerIndex_2D:
    def __init__(self, x, y) -> None:
        self._h = x
        self._k = y

    @property
    def h(self) -> int:
        return self._h

    @property
    def k(self) -> int:
        return self._k

    def __len__(self) -> float:
        return np.sqrt(self.h ** 2 + self.k ** 2)

    def __str__(self) -> str:
        return f"[{self.h}{self.k}]"

    def __add__(self, other) -> 'MillerIndex_2D':
        if isinstance(other, MillerIndex_2D):
            return MillerIndex_2D(self.h + other.h, self.k + other.k)
        elif isinstance(other, tuple) and len(other) == 2:
            return MillerIndex_2D(self.h + other[0], self.k + other[1])
        else:
            raise TypeError(f"{other} is not an appropriate Miller Index (2D)")


def peak_predict(diffraction: Diffraction2D, d_spacings: tuple[float, float]) -> object:
    """
    Predict the peak postion (q) from real space parameters
    :return: peak_locs: peak positions in q
    """
    peak_locs = []
    num_pixels = diffraction.num_pixels
    peak_locs_theor = sorted(np.round(np.divide(num_pixels, d_spacings)))

    centre = num_pixels // 2
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


if __name__ == "__main__":
    peak_predict()
