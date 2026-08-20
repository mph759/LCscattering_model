from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from enum import StrEnum

from utils import check_existing_ext, fix_file_ext

text_width = 6.693
text_height = 9.331


def textsize_scale(y_scale: float | int = 2 / 5, x_scale: float | int = 1) -> tuple[float, float]:
    x, y = (text_width, text_height)
    return x * x_scale, y * y_scale


def textsize_square(scale: float | int = 1) -> tuple[float, float]:
    return text_width * scale, text_width * scale

class AxesLabel(StrEnum):
    ANGLE = 'Angle (\u00B0)'
    ANGLE_SVG = r'Angle \$ \left( ^\circ \right) \$'
    THETA = r'$\Theta$ / $^\circ$'
    INTENSITY = r'Intensity (arb. units)'
    Q = r'q'
    Q_INV_NM = r'q / nm$^{-1}$'
    R = r'r'


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['savefig.format'] = 'svg'
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['figure.figsize'] = textsize_scale(2 / 5)
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlelocation'] = 'left'
#plt.rcParams['font.size'] = 16

def align_ylim(ax: plt.Axes, x_range=(0, 0), scale: float = 1.5, edge_mask: float = 0):
    line_data = [line.get_data()[1][x_range[0] + edge_mask: x_range[1] - edge_mask] for line in ax.get_lines()]
    min_line = np.min(line_data)
    max_line = np.max(line_data)
    del line_data

    y_min = scale * np.min(min_line)
    y_max = scale * np.max(max_line)
    ax.set_ylim(y_min, y_max)


def save(fig: plt.figure, array: np.ndarray, file_name: str, file_type: str = None, close_fig: bool = True,
         **kwargs) -> str:
    """
    Save the figure as a numpy file or as an image
    :param fig: Figure object to be saved
    :param array: numpy array to be saved
    :param file_name: Output file name
    :param file_type: Type of file you want to save (e.g. npy or jpg).
    :param close_fig: Boolean for whether to close the figure after saving.
    If not given, file name is checked for existing extension. Otherwise, default npy file
    :return:
    """
    if file_type is None:
        file_name, file_type = check_existing_ext(file_name)
        if file_type is None:
            file_type = "npy"
    file_name = fix_file_ext(file_name, file_type)
    if file_type == "npy":
        np.save(Path(file_name), array)
    else:
        try:
            fig.savefig(Path(file_name), format=file_type, **kwargs)
        except ValueError:
            raise ValueError(f"Format \'{file_type}\' is not supported (supported formats: npy, eps, jpeg, jpg, pdf, "
                             f"pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp)")
    if close_fig:
        plt.close(fig)
    return file_name
