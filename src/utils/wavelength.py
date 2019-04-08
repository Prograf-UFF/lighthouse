import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from operator import itemgetter
from src.utils.utils import show_image, WAVE_WINDOW_SIZE_WIDTH, WAVE_WINDOW_SIZE_HEIGHT, distance_euclidean
from ..utils.sobel import sobel_filter, sort_g, cdf, get_xy, filtrar_g


SAVGOL_WINDOW_SIZE = 197
PEAKS_MIN_DISTANCE = 400  # In pixels
GRAVITY = 9.80665 # In m/s^2
ONE_NUDO = 1852.0/3600.0 # meters per second


def binarized_dilate_image(image: np.ndarray, show_: bool=False, bin_cv2: bool=True) -> np.ndarray:
    ''' binarize the image and apply dilation so that the curves are not very thick
    :param image: image input in BGR
    :param show_: if is True show the binarization and dilate result
    :param bin_cv2:
    :return: binarized and dilated image
    '''

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if bin_cv2:
        _, image_bin = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #_, image_bin = cv2.threshold(image_gray, 70, 255, cv2.THRESH_BINARY)
    else:
        g = image_gray
        sort_g_tupla = sort_g(g, orden=True)
        acu_cdf = cdf(sort_g_tupla, False)

        rd = random.uniform(0.95, 0.98)
        xy = get_xy(rd, acu_cdf, sort_g_tupla)
        print("random #:", rd)

        image_bin = filtrar_g(g, g[xy[1], xy[0]], False)
    kernel = np.ones((3, 3), np.uint8)
    image_dilate = cv2.dilate(image_bin, kernel, iterations=2)
    if show_:
        show_image(image_dilate, "binarization", WAVE_WINDOW_SIZE_WIDTH, WAVE_WINDOW_SIZE_HEIGHT)
    return image_dilate

def get_wave_edge(image: np.ndarray, show_: bool=False) -> np.ndarray:
    ''' We obtain the coordinates represented by the wave
    :param image: image(ROI) in black and white
    :param show_: if is True show the edge wave
    :return: coordinates that represents the wave(: [(y1, x1), (y2, x2), ...])
    '''
    # we only create a matrix with zeros to then show the wave that will be extracted
    img_wave = np.zeros(image.shape, dtype=image.dtype)
    # here we store the coordinates of the extracted wave
    wave_coord = []
    # we scan the image from the lower left end upwards until we find a black pixel (representing the wave)
    # and consider it as part of the required wave
    for x in range(0, image.shape[1]):
        for y in range(image.shape[0] - 1, 0, -1):
            if image[y, x] == 0:
                img_wave[y, x] = 255
                wave_coord.append((y, x))
                break
            if y == 1 and x > 0:
                wave_coord.append((wave_coord[x - 1][0], x))
    if show_:
        show_image(img_wave, "waves-skeleton", WAVE_WINDOW_SIZE_WIDTH, WAVE_WINDOW_SIZE_HEIGHT)
    return wave_coord

def smooth_wave(wave_coord: np.ndarray, show_: bool=False) -> np.ndarray:
    ''' we use the 'savgol_filter' algorithm to smooth the wave
    :param wave_coord: List's tuples [(y,x), ...] that store wave's coordinates
    :param show_: if is True show the normal waves(color gray) and the smooth wave(color blue)
    :return: 'Y' coordinates smoothed
    '''
    y_ = [y for y, x in wave_coord]
    # we use the 'savgol_filter' algorithm to smooth the wave
    yhat = savgol_filter(y_, SAVGOL_WINDOW_SIZE, 3, mode='nearest')  # window size 197, polynomial order 3

    if show_:
        x_ = [x for y, x in wave_coord]
        #fig, ax = plt.subplot()
        plt.plot(y_, color='gray')
        plt.plot(x_, yhat, '--', color='blue')
        plt.legend(('original wave', 'smooth wave'), loc='lower center')
    return  yhat

def find_max_locals(wave_coord: np.ndarray, yhat: np.ndarray, show_: bool=False) -> np.ndarray:
    ''' We find the local maxima using the 'find_peaks' algorithm, with the condition that the local maxima
        have a distance > PEAKS_MIN_DISTANCE pixels
    :param wave_coord: List's tuples [(y,x), ...] that store wave's coordinates
    :param yhat: 'Y' coordinates smoothed
    :param show_: if is True show the peaks(local maxima, color red), two first peaks higher
    :return: ordered list of local maxima
    '''
    y_ = [y for y, x in wave_coord]
    # x_peaks are indices or location where the highest peaks are
    x_peaks, _ = find_peaks(yhat, distance=PEAKS_MIN_DISTANCE)
    y_peaks = [yhat[i] for i in x_peaks]

    # We arrange the coordinates in a descending way and we are left with the first two
    peaks_coords = [(y_peaks[i], x_peaks[i]) for i in range(0, len(x_peaks))]
    peaks_coords.sort( key=itemgetter(0), reverse=True)
    if show_:
        x_peaks_new = [x_new for y_new, x_new in peaks_coords[:2]]
        y_peaks_new = [y_new for y_new, x_new in peaks_coords[:2]]
        plt.plot(x_peaks_new, y_peaks_new, "x", color='red')
    return peaks_coords

def get_wavelength(peaks_coords: np.ndarray) -> float:
    '''
    :param peaks_coords: ordered list of local maxima
    :return: Euclidean distance between the two highest peaks
    '''
    x_ = [x for y, x in peaks_coords[:2]]
    y_ = [y for y, x in peaks_coords[:2]]
    wl = distance_euclidean([y_[0], x_[0]], [y_[1], x_[1]])
    return wl

def get_estimated_speed(wavelength: float) -> float:
    '''
    :param wavelength: wave length
    :return: speed in meters per second
    '''
    return math.sqrt((wavelength*GRAVITY)/(2.0*math.pi))

def ms_to_nudo(ms: float) -> float:
    '''
    :param ms: meters per secon
    :return: value in nudos
    '''
    return ms/ONE_NUDO