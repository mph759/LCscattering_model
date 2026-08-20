from matplotlib import pyplot as plt
from enum import StrEnum

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
