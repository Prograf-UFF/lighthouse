import sys
import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
import matplotlib.pyplot as plt
from ..utils.utils import *


class RectifyAffine:
    # ---------- Settings ---------
    nLinePairs = 2  # Select this may pairs of perpendicular lines

    def __init__(self, path, image):
        self.path = path
        self.image = image

    def get_perspective_transform(self):
        print("*************** START ****************")
        im = image_read(self.path, self.image)
        print("-Limits: ", im.shape)
        # Create figure
        replot_affine(im, im.shape)

        # Lines
        lines = [[], []]
        x = [[], []]
        y = [[], []]
        vPts = []

        # Get line pairs interactively
        for i in range(0, 2*self.nLinePairs):
            ii = i % 2
            if ii == 1:
                plt.suptitle('Click two points intersecting a line parallel to the green line')
            else:
                if i == 0:
                    plt.suptitle('Click two points intersecting the first of two parallel lines')
                else:
                    plt.suptitle(
                        'Click two points intersecting the first of two parallel lines not parallel to the first set')
            x1, y1, line = get_line()
            x[ii].append(x1)
            y[ii].append(y1)
            lines[ii].append(line)
            if ii == 1:
                nlp = len(lines[0])
                vPt = np.cross(lines[0][nlp - 1], lines[1][nlp - 1])
                if vPt[2] != 0.:
                    vPt[0] = vPt[0] / vPt[2]
                    vPt[1] = vPt[1] / vPt[2]
                    vPt[2] = vPt[2] / vPt[2]
                vPts.append(vPt)
            # re-plot figure
            replot_affine(im, im.shape, lines, x, y, vPts)
            print("~~~~~~~~~~~~~~~ " + str(i) + "th line pair ~~~~~~~~~~~~~~~~~~")
        # AQUI: usar print_array(), no en en replot_affine()
        vline = get_vanish_line(im.shape, vPts[0], vPts[1], draw_=True)

        print("*************** END ****************")
        show()


