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

        #draw_line(im.shape, lines[0][0], 'y-')
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
            H = scaleHToImage(H, im.shape, True)
            print_array("H Scaling", H)

        print_array("H FINAL", H)
        plt.figure(2)
        imRect = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
        imshow(imRect, cmap='gray')
        #plt.figure(2)
        #img_bgr = create_image(im.shape)
        #imshow(img_bgr, cmap='gray')

        x_p = np.dot(H, np.matrix(pts[0]).transpose())
        print("x_p:", x_p)
        print("***************************** END ******************************")
        show()


    def matrix_h(self):
        im = image_read(self.path, self.image)
        print("-Limits: ", im.shape)
        plt.close()  # ginput does not allow new points to be plotted
        imshow(im, cmap='gray')
        axis('image')
        ax = plt.gca()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        line = []
        x1 = []
        y1 = []

        # Create figure
        plt.figure(1)
        for i in range(0, 1):
            x1, y1, line = get_line()

        xx, yy = getPlotBoundsLine(im.shape, line)
        plot(xx, yy, 'r-')

        vLine = line
        H = np.identity(3)
        H[2, 0] = vLine[0]
        H[2, 1] = vLine[1]
        H[2, 2] = vLine[2]

        print_array("H", H)
        # Compute transformed line = H^-T * l
        #H_ = np.linalg.inv(H)
        #print_array("H_T", H_)

        pt = plt.ginput(n=1)
        c = [pt[0][0], pt[0][1], 1]
        #print(c)
        plt.plot(pt[0][0], pt[0][1], 'yo')

        pt2 = plt.ginput(n=1)
        d = [pt2[0][0], pt2[0][1], 1]
        # print(c)
        plt.plot(pt2[0][0], pt2[0][1], 'yo')

        pt3 = plt.ginput(n=1)
        e = [pt3[0][0], pt3[0][1], 1]
        # print(c)
        plt.plot(pt3[0][0], pt3[0][1], 'yo')

        c_ = np.dot(H, c)
        d_ = np.dot(H, d)
        e_ = np.dot(H, e)
        c_[0] = c_[0]/c_[2]
        c_[1] = c_[1]/c_[2]
        c_[2] = 1
        d_[0] = d_[0]/d_[2]
        d_[1] = d_[1]/d_[2]
        d_[2] = 1
        e_[0] = e_[0]/e_[2]
        e_[1] = e_[1]/e_[2]
        e_[2] = 1



        #d_ = np.dot(M, d)
        limit_x_ = 600
        limit_y_ = 600

        A = [[], [], [], [],[],[]]
        A[0] = [c_[0], c_[1], 1, 0, 0, 0, 0]
        A[1] = [0 , 0, 0, c_[0], c_[1], 1, 0]

        A[2] = [d_[0], d_[1], 1, 0, 0, 0, -(limit_x_-1)]
        A[3] = [0, 0, 0, d_[0], d_[1], 1, 0]

        A[4] = [e_[0], e_[1], 1, 0, 0, 0, 0]
        A[5] = [0, 0, 0, e_[0], e_[1], 1, -(limit_y_-1)]
        print_array("A", A)

        u,s, vh = np.linalg.svd(A)
        print_array("vh", vh)
        print_array("s", s)

        T = np.identity(3)
        T[0][0] = vh[6][0]
        T[0][1] = vh[6][1]
        T[0][2] = vh[6][2]
        T[1][0] = vh[6][3]
        T[1][1] = vh[6][4]
        T[1][2] = vh[6][5]
        T[2][2] = vh[6][6]

        print_array("*******T", T)
        print_array("Tc_:", np.dot(T, c_))
        d2 = np.dot(T, d_)
        e2 = np.dot(T, e_)
        print_array("Td_:", [d2[0]/d2[2], d2[1]/d2[2]])
        print_array("Te_:", [e2[0]/e2[2], e2[1]/e2[2]])


        M = np.dot(T, H)

        M_ = np.linalg.inv(M)
        res = np.dot(M_, [0,0,1])
        print_array("res c:", res/res[2])
        print("c:", c)

        res = np.dot(M_, [limit_x_-1, 0, 1])
        print_array("res d:", res / res[2])
        print("d:", d)

        res = np.dot(M_, [0, limit_y_ -1, 1])
        print_array("res e:", res / res[2])
        print("e:", e)

        limit_x = im.shape[1]
        limit_y = im.shape[0]
        plt.figure(2)

        im_result = create_image([limit_y_,limit_x_], im.dtype, im.ndim)
        for x_ in range(0, limit_x_):
            for y_ in range(0, limit_y_):
                p = np.dot(M_, [x_, y_, 1])
                if p[2]!=0:
                    x = int(np.round(p[0]/p[2]))
                    y = int(np.round(p[1]/p[2]))

                    if x>=0 and x<limit_x and y>=0 and y<limit_y:
                        im_result[y_,x_] = im[y,x]


        imshow(im_result)



        show()

