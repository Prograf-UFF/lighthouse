import sys
import numpy as np
from scipy import signal
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from operator import itemgetter
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2

def sobel_filter(path, image, k_size=3):
    im = cv2.imread(path + image, 0)
    img = im.astype('int32')

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
    #print(g[:3])
    g *= 255.0 / np.max(g)  # normalize (Q&D)
    print(g.shape)
    # plt.figure()
    plt.imshow(g, cmap='gray')
    plt.show()

    return g

def sort_g(g):
    get_tuplas = []
    for x in range(0, g.shape[1]):
        for y in range(0, g.shape[0]):
            get_tuplas.append((g[y, x] , [x, y]))

    get_tuplas.sort(key=itemgetter(0))
    return get_tuplas

def cdf(tuplas, show_=True):
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
        #print("maximo:", np.max(val_acumulado))
        plt.scatter(X, val_acumulado, color='yellowgreen', marker='.', label='cdf')
        plt.show()
    return val_acumulado

def get_xy(n_rand, cdf, sort_tupla):
    i = 0
    for x in range(0, len(cdf)):
        if cdf[x]>=n_rand:
            i = x
            break
    # get coord
    return sort_tupla[i][1]


def filtrar_g(g, val_limiar, show_=True):
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

# ----------------- SOBEL ----------------
def sobel(path, image):
    im = cv2.imread(path + image, 0) #scipy.misc.imread(path + image)
    im = im.astype('int32')
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)

    scipy.misc.imsave(path + '/sobel.jpg', mag)
    plt.imshow(mag, cmap='gray')
    plt.show()