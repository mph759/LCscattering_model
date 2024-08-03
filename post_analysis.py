import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from correlation import AngularCorrelation


if __name__ == '__main__':
    data_path = Path().cwd() / r'output\LCscattering-trial_2024-07-26 11-38-08\unit_vector_45\angular_corr.npy'
    angular_correlation = AngularCorrelation.load(data_path, num_r=4096, num_th=720)
    angular_correlation.plot(clim=5e11)
    angular_correlation.plot_line(1172)
    plt.show()

