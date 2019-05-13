import matplotlib.pyplot as plt
import numpy as np
from src.utils.ransac import ransac_base
import cv2
from typing import List, Tuple


# Window's size of the wave in cv2.imshow
WAVE_WINDOW_SIZE_WIDTH = 900
WAVE_WINDOW_SIZE_HEIGHT = 470

IMG_TO_GET_VLINE = 'src/images/base/img_to_get_vLine.jpg'
P1_XYW, P2_XYW = [271*4, 507*4, 1], [682*4, 558*4, 1]   # points taken manually
P3_XYW, P4_XYW = [858*4, 495*4, 1], [1015*4, 549*4, 1]  # points taken manually


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


