import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from math import pi
from pylab import plot, ginput, show, axis, imshow, draw
import numpy as np
import os
import pprint
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


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
    pts = np.around(pts)
    pts_h = [[x[0], x[1], 1] for x in pts]
    line = np.cross(pts_h[0], pts_h[1])  # line is [p0 p1 1] x [q0 q1 1]
    # return points that were clicked on for plotting
    x1 = map(lambda x: x[0], pts)  # map applies the function passed as
    y1 = map(lambda x: x[1], pts)  # first parameter to each element of pts
    return x1, y1, line


def draw_points(pts, color, prints=False):
    if prints:
        print("----------------  Intersection's Points:  --------------")
    for i in range(0, len(pts)):
        plot(pts[i][0], pts[i][1], color)
        if prints:
            print("pt(", i, "):", pts[i])
    if prints:
        print("-----------------------------------------------------")


def get_intersection_point(line1, line2):
    vPt = np.cross(line1, line2)
    if vPt[2] != 0.:
        vPt[0] = vPt[0] / vPt[2]
        vPt[1] = vPt[1] / vPt[2]
        vPt[2] = vPt[2] / vPt[2]
    return vPt


def print_array(label, list_):
    print("----------------  " + label + ":  --------------")
    pprint.pprint(list_)
    print("-----------------------------------------------------")


def draw_line(limits, line, color):
    xx, yy = getPlotBoundsLine(limits, line)
    plot(xx, yy, color)


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

    # Limit axes to the image
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])


# INPUTS:
#   H      = projective transformation
#   line   = line in the source image (domain of H)
# OUTPUTS:
#   HR * H = New H where HR is a rotation chosen to make line map to
#            either vertical or horizontal, chosen
def rotateHToLine(H, line):
    assert len(line) == 3
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Compute transformed line = H^-T * l
    H_T = np.linalg.inv(H).T
    lineTr = np.dot(H_T, line)
    print_array("H_T", H_T )
    print_array("lineTr", lineTr )

    # Rotate so that this line is horizonal in the image
    r1 = np.array([lineTr[1], -lineTr[0]]) # First row of R is perpendicular to linesTr[0]
    r1 = r1 / np.linalg.norm(r1.flatten(1))
    theta = np.arctan2(-r1[1] , r1[0])
    if abs(theta) < pi/4:
        R = np.array([[r1[0],  r1[1]], [-r1[1], r1[0]]])
    else:
        R = np.identity(2)
        # R = np.array([[r1[1], -r1[0]], [ r1[0], r1[1]]])
    theta = np.arctan2(R[1,0], R[1,1])
    print("Rotating by %.1f degrees" % (theta*180/pi))
    HR = np.identity(3)
    HR[0:2,0:2] = R

    return np.dot(HR,H)


# INPUTS:
#   H      = perspective transformation (3x3 matrix)
#   limits = image boundaries
# OUTPUTS:
#   Htr,Hbr,Hbl = homogeneous 3-vectors = H * (image corners)
def getHCorners(H, limits):
    Ny = float(limits[0])
    Nx = float(limits[1])
    print("H:", H, "asdads:",np.array([0.0, Ny, 1.0]).flatten(1))
    # Apply H to corners of the image to determine bounds
    Htr = np.dot(H, np.array([0.0, Ny, 1.0]).flatten(1))  # Top left maps to here
    print("Htr", Htr, )
    Hbr = np.dot(H, np.array([Nx, Ny, 1.0]).flatten(1))  # Bottom right maps to here
    Hbl = np.dot(H, np.array([Nx, 0.0, 1.0]).flatten(1))  # Bottom left maps to here
    Hcor = [Htr, Hbr, Hbl]

    # Check if corners in the transformed image map to infinity finite
    finite = True
    for y in Hcor:
        if y[2] == 0:
            finite = False

    print_array("Hcor",Hcor)
    return Hcor, finite


# INPUTS:
#   H      = projective transformation (3x3)
#   limits = image size
# OUTPUTS:
#   HT*H   = new projective transformation such that HT is a translation and
#            HT*H x > 0 for all x > 0
def translateHToPosQuadrant(H, limits):
    assert len(limits) >= 2 # can have color channels
    assert limits[0] > 0 and limits[1] > 0
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Get H * image corners
    Hcor, finite = getHCorners(H, limits)

    # Check if corners map to infinity, if so skip translation
    if not finite:
        print("Corners map to infinity, skipping translation")
        return H

    # Min coordinates of H * image corners
    minc = [min([Hcor[j][i]/Hcor[j][2] for j in range(len(Hcor))]) for i in range(2)]
    print_array("minc", minc)
    # Choose translation
    HT = np.identity(3)
    HT[0,2] = -minc[0]
    HT[1,2] = -minc[1]

    return np.dot(HT, H)


# INPUTS:
#   H      = perspective transformation (3x3 matrix)
#   limits = image boundaries
# OUTPUTS:
#   HS * H, where HS is an (isotropic) scaling to keep an image of shape
#   limits contained within limits when HS*H is applied
def scaleHToImage(H, limits, anisotropic=False):  # TODO: test anisotropic
    assert len(limits) >= 2  # can have color channels
    assert limits[0] > 0 and limits[1] > 0
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Get H * image corners
    Hcor, finite = getHCorners(H, limits)

    # If corners in the transformed image are not finite, don't do scaling
    if not finite:
        print("Skipping scaling due to point mapped to infinity")
        return H;

    # Maximum coordinate that any corner maps to
    k = [max([Hcor[j][i] / Hcor[j][2] for j in range(len(Hcor))]) / float(limits[1 - i]) for i in range(2)];
    print_array("k", k)

    # Scale
    if anisotropic:
        print("Scaling by (%f,%f)\n" % (k[0], k[1]))
        HS = np.array([[1. / k[0], 0.0, 0.0], [0.0, 1. / k[1], 0.0], [0.0, 0.0, 1.0]])
    else:
        k = max(k)
        print("Scaling by %f\n" % k)
        HS = np.array([[1.0 / k, 0.0, 0.0], [0.0, 1.0 / k, 0.0], [0.0, 0.0, 1.0]])
    print_array("HS", HS)
    return np.dot(HS, H)


def create_image(shape, dtype, nchanels):
    img = np.zeros([shape[0], shape[1], nchanels], dtype=dtype)
    r, g, b = cv2.split(img)
    img_bgr = cv2.merge([b, g, r])
    return img_bgr