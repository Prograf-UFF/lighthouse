from src.rectifyAffine.rectifyAffine import RectifyAffine
import numpy as np
from src.utils.utils import *
from src.utils.ransac import *
from src.utils.canny import *

if __name__ == "__main__":
    path = 'src/images/'
    img = 'exemplo3.jpg'

    #rectAffine = RectifyAffine(path=path, image=img)
    #rectAffine.matrix_h()

    im = image_read(path, img)

    #im_r = single_band_representation(im)
    #main_canny(path, img)
    detect_border = get_canny(im)

    #auto_canny_(path, img)

    # main_ransac()
    #get_quadrante(im)

    sx, sy = get_samples(detect_border)
    line_X, line_y_ransac = ransac(sx, sy)
    draw_line_ransac(line_X, line_y_ransac, im)

    print("Finished...!")

