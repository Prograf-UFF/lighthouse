import sys
import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
import matplotlib.pyplot as plt
from ..utils.utils import *
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2

class RectifyAffine:
    # ---------- Settings ---------
    nLinePairs = 2  # Select this may pairs of perpendicular lines

    def __init__(self, path, image):
        self.path = path
        self.image = image

    def get_perspective_transform(self, doRotationAfterH=True, doTranslationAfterH=True, doScalingAfterH=True):
        print("***************************** START ******************************")
        im = image_read(self.path, self.image)
        print("-Limits: ", im.shape)
        # Create figure
        plt.figure(1)
        replot_affine(im, im.shape)

        # Lines
        lines = [[], []]
        x = [[], []]
        y = [[], []]
        vPts = [] # vanish pints
        pts = [[], [], [], []] # intersection's points

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
        print_array("lines: ", lines)
        print_array("vanish Points(vPts)", vPts)
        vLine = get_vanish_line(im.shape, vPts[0], vPts[1], draw_=True)
        print_array("Vanish Line", vLine)

        H = np.identity(3)
        H[2, 0] = vLine[0] / vLine[2]
        H[2, 1] = vLine[1] / vLine[2]

        print_array("H", H)
        # 4th points intersection's lines
        pts[0] = get_intersection_point(lines[0][0], lines[1][1])
        pts[1] = get_intersection_point(lines[1][0], lines[1][1])
        pts[2] = get_intersection_point(lines[0][0], lines[0][1])
        pts[3] = get_intersection_point(lines[1][0], lines[0][1])
        draw_points(pts, 'ro', True)

        # Rotate after doing H
        if doRotationAfterH:
            H = rotateHToLine(H, lines[0][0])
            print_array("H Rotation", H)

        # Translate to keep Hx > 0
        if doTranslationAfterH:
            H = translateHToPosQuadrant(H, im.shape)
            print_array("H Translation", H)

        # Scale to keep the output contained just within the image matrix
        if doScalingAfterH:
            H = scaleHToImage(H, im.shape, False)
            print_array("H Scaling", H)

        print_array("H FINAL", H)

        imRect = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
        imshow(imRect, cmap='gray')
        #plt.figure(2)
        img_bgr = create_image(im.shape)
        #imshow(img_bgr, cmap='gray')
        print("***************************** END ******************************")
        show()



