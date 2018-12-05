import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
import numpy as np


def get_canny(img_gray: np.ndarray, threshold: int=45) -> np.ndarray:
    """ edged detector.
    :param img_gray: gray scale image.
    :param threshold: the threshold.
    :return: image with the edges detected.
    """
    low_threshold = threshold
    max_low_threshold = threshold*3
    kernel_size = 3
    # Reduce noise with a kernel 3x3
    detected_edges = cv2.blur(img_gray, (3, 3))

    # Canny detector
    detected_edges = cv2.Canny(detected_edges, low_threshold, max_low_threshold, kernel_size)
    return detected_edges


def auto_canny_main(image: np.ndarray, sigma: float=0.33) -> np.ndarray:
    """ detect edges without the need for a threshold.
    :param image: the input image.
    :param sigma: can be used to vary the percentage thresholds that are determined based on simple statistics.
    :return: image with the edges detected.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def auto_canny(image) -> np.ndarray:
    """ detect edges without the need for a threshold.
    :param image: the input image.
    :return: image with the edges detected.
    """
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    return auto_canny_main(blurred)