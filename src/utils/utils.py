import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
import numpy as np
import os
import pprint


def image_read(path, image):
    img_path = path + image
    if os.path.exists(img_path):
        img = mplimg.imread(img_path)
        return img
    else:
        raise Exception("image not exist...!")


# Get two mouse clicks from the user with ginput to select points
# return coordinates of clicked points (x1[i],y1[i]) and line through the
# pair of points, a non-normalized (homogeneous) 3-vector
def get_line():
    # get mouse clicks
    pts = []
    while len(pts) == 0:  # FIXME
        pts = plt.ginput(n=2)
    # pts = np.around(pts)
    pts_h = [[x[0], x[1], 1] for x in pts]
    line = np.cross(pts_h[0], pts_h[1])  # line is [p0 p1 1] x [q0 q1 1]
    # return points that were clicked on for plotting
    x1 = map(lambda x: x[0], pts)  # map applies the function passed as
    y1 = map(lambda x: x[1], pts)  # first parameter to each element of pts
    return x1, y1, line


def draw_points(x, y, color, prints=False):
    x1 = [a for a in list(x)]
    y1 = [b for b in list(y)]
    if prints:
        print("Xs: ", x1)
        print("Ys: ", y1)
    # 'color=g^' is a color => g=green and ^ is a form of the point
    plt.plot(x1, y1, color)


def print_array(label, list_):
    print("----------------  " + label + ":  --------------")
    pprint.pprint(list_)
    print("-----------------------------------------------------")


def get_vanish_line(limits, p1, p2, draw_=True):
    line = np.cross(p1, p2)  # line is [p0 p1 1] x [q0 q1 1]
    if draw_:
        xx, yy = getPlotBoundsLine(limits, line)
        plot(xx, yy, 'r-')
    return line

# INPUT:
#   size = image size
#   l    = line (homogeneous 3-vector)
# OUTPUTS:
#   xx,yy = plot(xx,yy) will plot the line cropped within the image region
def getPlotBoundsLine(size, l):
    l = l.flatten(1)
    L = 0
    R = 1
    T = 2
    B = 3
    Nx = size[1]
    Ny = size[0]
    # lines intersecting image edges
    lbd = [[] for x in range(4)]
    lbd[L] = np.array([1.0, 0.0, 0.0])
    lbd[R] = np.array([1.0, 0.0, -float(Nx)])
    lbd[T] = np.array([0.0, 1.0, 0.0])
    lbd[B] = np.array([0.0, 1.0, -float(Ny)])
    I = [np.cross(l, l2) for l2 in lbd]

    # return T/F if intersection point I is in the bounds of the image
    Ied = [] # List of (x,y) where (x,y) is an intersection of the line with the boundary
    for i in [L, R]:
        if I[i][2] != 0:
            In1 = I[i][1] / I[i][2]
            if In1 > 0 and In1 < Ny:
                Ied.append(I[i][0:2]/I[i][2])

    for i in [T, B]:
        if I[i][2] != 0:
            In0 = I[i][0] / I[i][2]
            if In0 > 0 and In0 < Nx:
                Ied.append(I[i][0:2]/I[i][2])

    assert(len(Ied) == 2 or len(Ied) == 0)
    xx = [Ied[x][0] for x in range(0,len(Ied))]
    yy = [Ied[x][1] for x in range(0,len(Ied))]

    return xx, yy


# --------- Supporting functions -----------
# Plot image, lines, vanishing points, vanishing line
def replot_affine(im, limits, lines=[[], []], x=[[], []], y=[[], []], vPts=[]):
    # -- Settings for this function ---
    plot_lines = True
    plot_vpts = True
    plot_points = True
    plot_vline = True
    # ---------------------------------
    plt.close()  # ginput does not allow new points to be plotted
    imshow(im, cmap='gray')
    axis('image')
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Determine how many lines to plot in red, leaving the last in green if the second needs to be picked
    nl1 = len(y[0])
    nl2 = len(y[1])
    if nl1 == nl2:
        nred = nl1
    else:
        nred = nl1 - 1
    # Plot extension of user-selected lines (dashed)
    if plot_lines:
        for k in range(nred):
            xx, yy = getPlotBoundsLine(limits, lines[0][k])
            plot(xx, yy, 'w:')
        if nl1 - nred > 0:
            xx, yy = getPlotBoundsLine(limits, lines[0][nl1 - 1])
            plot(xx, yy, 'g:')
            # print("line?:", xx, yy)
        for l in lines[1]:
            xx, yy = getPlotBoundsLine(limits, l)
            plot(xx, yy, 'k:')

    # Compute normalized vanishing points for plotting
    vPts_n = [[0, 0] for x in vPts]
    vPtInImage = [True for x in vPts]
    for i in range(len(vPts)):
        if vPts[i][2] == 0:
            vPtInImage[i] = False
        else:
            vPts_n[i][0] = vPts[i][0] / vPts[i][2]
            vPts_n[i][1] = vPts[i][1] / vPts[i][2]
            vPtInImage[i] = vPts_n[i][0] < limits[0] and vPts_n[i][0] > 0 and vPts_n[i][1] < limits[1] and vPts_n[i][1] > 0

    # Plot vanishing points
    if plot_vpts:
        for i in range(len(vPts_n)):
            if vPtInImage[i]:
                plot(vPts_n[i][0], vPts_n[i][1], 'yo')

    print_array("lines: ", lines)
    print_array("vanish Points(vPts)", vPts)

