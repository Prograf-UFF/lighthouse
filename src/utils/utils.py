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
#sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
from typing import List, Tuple


# Window's size of the wave in cv2.imshow
WAVE_WINDOW_SIZE_WIDTH = 900
WAVE_WINDOW_SIZE_HEIGHT = 470

IMG_TO_GET_VLINE = 'src/images/base/img_to_get_vLine.jpg'
P1_XYW, P2_XYW = [271*4, 507*4, 1], [682*4, 558*4, 1]   # points taken manually
P3_XYW, P4_XYW = [858*4, 495*4, 1], [1015*4, 549*4, 1]  # points taken manually


def my_print(headers: List[str], matrix: np.ndarray, title: str="") -> None:
    """ beautiful matrix print
    :param headers: matrix's headers
    :param matrix: matrix to print
    :param title: matrix's title
    :return: None, only print title, headers and matrix
    """
    cols_align = []
    cols_m = matrix.shape[1]
    rows_m = matrix.shape[0]
    for i in range(0, cols_m):
        cols_align.append("l") if i == 0 else cols_align.append("r")
    content = []
    headers = [chr(x) for x in range(97, 97 + cols_m)] if headers is None else headers
    content.append(headers)
    for i in range(0, rows_m):
        content.append(matrix[i])

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_header_align(cols_align)
    table.set_cols_dtype(['a']*cols_m)  # automatic
    table.set_cols_align(cols_align)
    table.add_rows(content)
    print("********************  " + title + "  *********************")
    print(table.draw())


def show_image(image: np.ndarray, title: str='title image', resize_w: int=1200, resize_h: int=800) -> None:
    """ Show image in OpenCV
    :param image: the input image
    :param title: image's title
    :param resize_w: window width
    :param resize_h: window height
    :return: None, only show image
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, resize_w, resize_h)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def show_image_properties(img: np.ndarray) -> None:
    """ show image properties.
    :param img: the input image
    :return: printing image properties
    """
    header = ["properties", "values"]
    channels = 1 if len(img.shape) < 3 else img.shape[2]
    content = [["width", img.shape[1]],
                    ["height", img.shape[0]],
                    ["channels", channels],
                    ["# of pixels", img.size],
                    ["data type", img.dtype]]
    my_print(header, np.array(content), "Image Property")


def distance_euclidean(v1: np.ndarray, v2: np.ndarray) -> float:
    """ get Euclidian distance of two points
    :param v1: (v1_x, v1_y) point
    :param v2: (v2_x, v2_y) point
    :return: distance between points v1 and v2
    """
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))


def get_quadrante() -> Tuple[np.ndarray, np.ndarray]:
    """ get two points, the first point is from the upper left corner
    and the second from the lower right corner of a quadrant
    :return: (pt1_x, pt1_y, 1), (pt2_x, pt2_y, 1), 1 is a homogeneous coordinate
    """
    pt1 = plt.ginput(n=1)
    pt2 = plt.ginput(n=1)

    pt1 = [pt1[0][0], pt1[0][1], 1]
    pt2 = [pt2[0][0], pt2[0][1], 1]
    return pt1, pt2


def fill_samples(p1: List, p2: List, s_x: List, s_y: List, img_canny: np.ndarray) -> None:
    """ Fill coordinates samples
    :param p1: (pt1_x, pt1_y, 1) point
    :param p2: (pt2_x, pt2_y, 1) point
    :param s_x: [] , empty array where we store coordinates 'x' to the sample
    :param s_y: [] , empty array where we store coordinates 'y' to the sample
    :param img_canny: image's edge
    :return: none, only s_x and s_y are filling
    """
    for x in range(int(p1[0]), int(p2[0])):
        for y in range(int(p1[1]), int(p2[1])):
            if img_canny[y, x] != 0:
                s_x.append([x])
                s_y.append(y)


def get_Vline_ransac(line_x: np.ndarray, line_y: np.ndarray) -> np.ndarray:
    """  get vanish line of RANSAC result
    :param line_x: coordinates that represent the x axis
    :param line_y: coordinates that represent the y axis
    :return: (x, y, z), the array with the coefficients of the vanishing line, in image coordinates.
    """
    plt.clf()
    plt.close('all')
    # get vanishLine
    p1 = [line_x[0][0], line_y[0], 1]  # begin of the line
    p2 = [line_x[len(line_y)-1][0], line_y[len(line_y)-1], 1]  # end of the line

    return np.cross(p1, p2)


def get_vanishLine_automatic() -> np.ndarray:
    """ with four points we calculate the vanish line
    :return: the array with the coefficients of the vanishing line, in image coordinates.
    """
    img_canny = cv2.imread(IMG_TO_GET_VLINE, 0)
    samples_x = []
    samples_y = []
    fill_samples(P1_XYW, P2_XYW, samples_x, samples_y, img_canny)
    fill_samples(P3_XYW, P4_XYW, samples_x, samples_y, img_canny)
    line_x, line_y_ransac, _ = ransac_base(np.array(samples_x), np.array(samples_y), show_=False)

    return get_Vline_ransac(line_x, line_y_ransac)


# ---------------- GET DIRECTION --------------
def get_direction(img: np.ndarray, img_orig: np.ndarray, get_q_auto: bool=True, show_result: bool=True, show_ransac: bool=True) -> None :
    """
    :param img: rectified image
    :param img_orig: original image, the input image
    :param get_q_auto: if it is true, the quadrant is the whole part of the rectified image
    :param show_result: plot two lines, lines that represent the trail
    :param show_ransac: plot RANSAC result
    :return: None
    """
    if not get_q_auto:
        plt.imshow(img, cmap='gray')
        plt.suptitle('Clique dois pontos para pegar a linha "V"')
        p1, p2 = get_quadrante()
    else:
        p1, p2 = [0.0, 0.0], [img.shape[1], img.shape[0]]
    samples_x = []
    samples_y = []
    fill_samples(p1, p2, samples_x, samples_y, img)

    line_X_pri, line_y_ransac_pri, inliers = ransac_base(np.array(samples_x), np.array(samples_y), show_=show_ransac, figure_num=7)


    return
    # segunda linea
    # calcular outliers que pertenecen a samples
    '''new_sx = []
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

        plt.show()'''
