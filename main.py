from src.rectifyAffine.rectifyAffine import RectifyAffine

import numpy as np
from src.utils.utils import *
from src.utils.ransac import *
from src.utils.canny import *
from src.utils.otsu import *
from src.utils.sobel import *
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

from src.proba.test import *

if __name__ == "__main__":
    path = 'src/images/images_bordas/'
    img = 'DSC_000016608'

    test_func(path=path, image_name=img)

    # rectAffine = RectifyAffine(path=path, image_name=img)
    # img_result = rectAffine.image_rectification(show_=True)

    # rectAffine.estimated_speed(img_result)
    # rectAffine.get_lineV(img_result, auto=False, sobel_show=False, show_filtrar_g=False, show_ransac=False)
    print("Finished...!")

