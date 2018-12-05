import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from operator import itemgetter
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
import copy


def sobel_filter(image: np.ndarray, k_size: int=3, show_: bool=True) -> np.ndarray:
    """ get image's edges
    :param image: the input image
    :param k_size: kernel-sobel size
    :param show_: plot edges
    :return: image's edges, represented as gradient values
    """
    img = copy.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype('int32')

    if k_size == 3:
        kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
        kv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)
    else:
        kh = np.array([[-1, -2, 0, 2, 1],
                       [-4, -8, 0, 8, 4],
                       [-6, -12, 0, 12, 6],
                       [-4, -8, 0, 8, 4],
                       [-1, -2, 0, 2, 1]], dtype=np.float)
        kv = np.array([[1, 4, 6, 4, 1],
                       [2, 8, 12, 8, 2],
                       [0, 0, 0, 0, 0],
                       [-2, -8, -12, -8, -2],
                       [-1, -4, -6, -4, -1]], dtype=np.float)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    gx = signal.convolve2d(img, kh, mode='same', boundary='symm', fillvalue=0)
    gy = signal.convolve2d(img, kv, mode='same', boundary='symm', fillvalue=0)

    g = np.sqrt(gx * gx + gy * gy)  # magnitude
    g *= 255.0 / np.max(g)  # normalize (Q&D)
    if show_:
        print(g.shape)
        plt.imshow(g, cmap='gray')
        plt.show()
    return g


def sort_g(g: np.ndarray) -> np.ndarray:
    """we order the values of the gradient ascending without losing the location they represent in the image
    :param g: image's edges, represented as gradient values
    :return: [[(g_x, g_y), (x, y)]] , value of the gradient and the position it represents
     in the input image sorted according to the value of the gradient ascending
    """
    get_tuplas = []
    for x in range(0, g.shape[1]):
        for y in range(0, g.shape[0]):
            get_tuplas.append((g[y, x], [x, y]))

    get_tuplas.sort(key=itemgetter(0))
    return get_tuplas


def cdf(tuplas: np.ndarray, show_: bool=True) -> np.ndarray:
    """ accumulated value of the gradients
    :param tuplas: [[(g_x, g_y), (x, y)]] value of the gradient and its positions in the image, ordered ascending
    :param show_: plot accumulated value
    :return: a list with the cumulative values of the gradients, minimum 0 and maximum 1
    """
    val_acumulado = []  # y=f(x)
    anterior = 0
    for x in range(0, len(tuplas)):
        val_acumulado.append(tuplas[x][0] + anterior)
        anterior = val_acumulado[len(val_acumulado)-1]
    # normalizar a 1
    val_max = np.max(val_acumulado)
    val_acumulado /= val_max
    if show_:
        X = [[i] for i in range(0, len(val_acumulado))]
        plt.scatter(X, val_acumulado, color='yellowgreen', marker='.', label='cdf')
        plt.show()
    return val_acumulado


def get_xy(n_rand: float, cdf: np.ndarray, sort_tupla: np.ndarray) -> np.ndarray:
    """
    :param n_rand: random number
    :param cdf: tuplas with accumulated values of gradients
    :param sort_tupla: sort gradients's values
    :return: position x, y that represent the position of the image, which will serve to obtain its value from the gradient
    """
    i = 0
    for x in range(0, len(cdf)):
        if cdf[x]>=n_rand:
            i = x
            break
    # get coord
    return sort_tupla[i][1]


def filtrar_g(g: np.ndarray, val_limiar: int, show_: bool=True) -> np.ndarray:
    """ only values higher than the threshold will be displayed
    :param g: image with values of gradients
    :param val_limiar: threshold
    :param show_: plot image's edge and image's edge filtered
    :return: image's edge filtered,
    """
    copy_g = np.zeros(g.shape, dtype=g.dtype)
    for x in range(0, g.shape[1]):
        for y in range(0, g.shape[0]):
            if g[y, x]<val_limiar:
                g[y, x] = 0
            if g[y, x]>=val_limiar:
                copy_g[y, x]= 255
    if show_:
        plt.figure(1)
        plt.imshow(copy_g, cmap='gray')
        plt.figure(2)
        plt.imshow(g, cmap='gray')
        plt.show()
    return copy_g

