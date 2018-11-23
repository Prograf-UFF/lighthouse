import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from math import pi
from pylab import plot, ginput, show, axis, imshow, draw
import numpy as np
import os
import pprint
import sys
from texttable import Texttable
from src.utils.ransac import *
from src.utils.otsu import *
from src.utils.canny import *
import copy

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


def get_vanish_line(limits, draw_=True):
    plt.suptitle('Clique dois pontos para gerar a "Vanish Line"')
    _, _, line = get_line()
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



def create_image(shape, dtype, nchanels):
    img = np.zeros([shape[0], shape[1], nchanels], dtype=dtype)
    r, g, b = cv2.split(img)
    img_bgr = cv2.merge([b, g, r])
    return img_bgr


def my_print(headers, matrix, title=""):
    cols_align = []
    cols_m = matrix.shape[1]
    rows_m = matrix.shape[0]
    for i in range(0, cols_m):
        if i == 0:
            cols_align.append("l")
        else:
            cols_align.append("r")

    content = []
    if headers == []:
        headers = [chr(x) for x in range(97, 97 + cols_m)]
    content.append(headers)
    for i in range(0, rows_m):
        content.append(matrix[i])

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_header_align(cols_align)
    table.set_cols_dtype(['a']*cols_m)  # automatic
    table.set_cols_align(cols_align)
    table.add_rows(content)

    if title != "":
        print("**********************  " + title + "  **********************")
    print(table.draw())


def show_image_properties(img):
    shape = img.shape
    header = ["properties", "values"]
    content = [["width", shape[1]],
                    ["height", shape[0]],
                    ["channels", shape[2]],
                    ["# of pixels", img.size],
                    ["data type", img.dtype]]
    my_print(header, np.array(content), "Image Property")


def get_point(draw_=True, cor='yo'):
    pt = plt.ginput(n=1)
    c = [pt[0][0], pt[0][1], 1]
    if draw_:
        plt.plot(pt[0][0], pt[0][1], cor)
    return c


def get_quadrante(img, draw_=True):
    p1 = get_point()
    p2 = get_point()

    p3 = [p2[0], p1[1]]
    p4 = [p1[0], p2[1]]
    if draw_:
        plt.plot(p3[0], p3[1], 'ro')
        plt.plot(p4[0], p4[1], 'ro')
        # print(p1, p2, p3, p4)
    return p1, p2


def add_samples(p1, p2, s_x, s_y, img_canny):
    x_ini = p1[0]
    y_ini = p1[1]
    x_fin = p2[0]
    y_fin = p2[1]
    for x in range(int(x_ini), int(x_fin)):
        for y in range(int(y_ini), int(y_fin)):
            if img_canny[y, x] != 0:
                s_x.append([x])
                s_y.append(y)


def get_samples(img_canny, show_scatter=False):
    p1, p2 = get_quadrante(img_canny)
    p3, p4 = get_quadrante(img_canny)
    plt.suptitle('Wait ..."')

    samples_x = []
    samples_y = []
    add_samples(p1, p2, samples_x, samples_y, img_canny)
    add_samples(p3, p4, samples_x, samples_y, img_canny)

    if show_scatter:
        plt.figure(2)
        plt.scatter(samples_x, samples_y, label='samples', color='k')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    return np.array(samples_x), np.array(samples_y)


def draw_line_ransac(line_x, line_y, img):
    # plt.plot(line_x, line_y, color='red', linewidth=2, label='RANSAC regressor')

    # get vanishLine
    p1 = [line_x[0][0], line_y[0], 1]
    p2 = [line_x[len(line_y)-1][0], line_y[len(line_y)-1], 1]

    vLine = np.cross(p1, p2)

    xx, yy = getPlotBoundsLine(img.shape, vLine)
    plot(xx, yy, 'g-', linewidth=1)
    return vLine


def get_vanishLine_manual(img, show_samples=False, show_ransac=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect_border = get_canny(img_gray)
    sx, sy = get_samples(detect_border, show_scatter=show_samples)
    line_X, line_y_ransac, _ = ransac(sx, sy, show_=show_ransac)

    plt.figure(1)
    # plt.cla()  # Clear axis
    plt.clf()  # Clear figure
    # plt.close()  # Close a figure window
    imshow(img, cmap='gray')
    axis('image')
    vLine = draw_line_ransac(line_X, line_y_ransac, img)
    return vLine


def get_vanishLine_automatic(img, image_base_path="src/images/base/", img_vline="img_to_get_vLine.jpg"):
    img_canny = cv2.imread(image_base_path + img_vline, 0)
    # puntos sacados manualmente
    p1, p2 = [271*4, 507*4, 1], [682*4, 558*4, 1]
    p3, p4 = [858*4, 495*4, 1], [1015*4, 549*4, 1]
    samples_x = []
    samples_y = []
    add_samples(p1, p2, samples_x, samples_y, img_canny)
    add_samples(p3, p4, samples_x, samples_y, img_canny)
    line_X, line_y_ransac, _ = ransac(np.array(samples_x), np.array(samples_y), show_=False)

    vLine = draw_line_ransac(line_X, line_y_ransac, img)
    return vLine

def distance_euclidean(v1, v2):
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))


def img_grayscale(img_rgb):
    r, g, b = cv2.split(img_rgb)
    image_bgr = cv2.merge([b, g, r])
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray_image


def negativo_grises(im):
    im_gray = img_grayscale(im)
    im5 = create_image(im.shape, im.dtype, 3)
    i = 0
    while i < im_gray.shape[0]:
        j = 0
        while j < im_gray.shape[1]:
            gris = im_gray[i,j]  # como es gris no importa
            valor = 255 - gris
            im5[i, j] = [valor, valor, valor]
            j+=1
        i+=1
    return im5

def use_erode_dilate(image):
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    erode = cv2.erode(image, kernel_erode)
    dilate = cv2.dilate(erode, kernel_dilate)

    return dilate

# --- AQUI SEGMENTAMOS EL CIELO --------
def segment_sky(image_base_path, img, img_save_neg="negative.jpg", img_save_otsu="/segment_sky.jpg"):
    my_img = image_read(image_base_path, img)
    neg_cor = negativo_grises(my_img)
    plt.imsave(image_base_path + "/" + img_save_neg, neg_cor)
    result = otsu(image_base_path, img_save_neg)
    erode_dilate = use_erode_dilate(result)
    cv2.imwrite(image_base_path + img_save_otsu, erode_dilate)
    cv2.imshow("sky", erode_dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_to_get_vLine(image_base_path, img_sky="segment_sky.jpg"):
    img_gray = cv2.imread(image_base_path + img_sky, 0)
    result = get_canny(img_gray, 60)
    cv2.imwrite(image_base_path + "/img_to_get_vLine.jpg", result)

# ---------------- GET DIRECTION --------------
def get_direction(img, img_orig, get_q_auto=True, show_result=True, show_ransac=True):
    if not get_q_auto:
        plt.imshow(img, cmap='gray')
        plt.suptitle('Clique dois pontos para pegar a linha "V"')
        p1, p2 = get_quadrante(img)
    else:
        p1, p2 = [0.0, 0.0],[img.shape[1], img.shape[0]]
    samples_x = []
    samples_y = []
    add_samples(p1, p2, samples_x, samples_y, img)

    if False:
        plt.figure(2)
        plt.scatter(samples_x, samples_y, label='samples', color='k')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    line_X_pri, line_y_ransac_pri, outliers = ransac(np.array(samples_x), np.array(samples_y), show_=show_ransac, figure_num=7)
    print(len(outliers), len(samples_x))

    # segunda linea
    # calcular outliers que pertenecen a samples
    new_sx = []
    new_sy = []
    for i in range(0, len(outliers)):
        if outliers[i]:
            new_sx.append(samples_x[i])
            new_sy.append(samples_y[i])
    line_X_sec, line_y_ransac_sec, _ = ransac(np.array(new_sx), np.array(new_sy), show_=show_ransac, figure_num=8)

    if show_result:
        plt.figure(9)
        img_original = copy.copy(img_orig)
        plt.imshow(img_original, cmap='gray')

        plt.plot(line_X_pri, line_y_ransac_pri, color='red', linewidth=1)
        plt.plot(line_X_sec, line_y_ransac_sec, color='green', linewidth=1)

        plt.show()
