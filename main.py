from src.rectifyAffine.rectifyAffine import RectifyAffine
import numpy as np
from src.utils.utils import *
from src.utils.ransac import *
from src.utils.canny import *
from src.utils.otsu import *
from src.utils.sobel import *
import random

if __name__ == "__main__":
    path = 'src/images/'
    img = 'exemplo5.jpg'
    image_base_path = 'src/images/base/'
    img_base = "DSC_000009279.jpg"

    rectAffine = RectifyAffine(path=path, image=img)
    #rectAffine.matrix_h()

    # 1º
    # segment_sky(image_base_path, img_base)
    # 2º
    # img_to_get_vLine(image_base_path, "segment_sky.jpg")

    #main_canny('src/images/save_images/', img, l_limiar=60)
    #sobel('src/images/save_images/', img)
    g = sobel_filter('src/images/save_images/', img, 3)
    sort_g_tupla = sort_g(g)
    acu_cdf = cdf(sort_g_tupla, False)

    #otsu('src/images/save_images/', "sobel.jpg", True)
    #main_canny('src/images/save_images/', "sobel.jpg", l_limiar=60)

    rd = random.uniform(0.75, 0.9)
    xy = get_xy(rd, acu_cdf, sort_g_tupla)

    print("random #:", rd)
    print("g max val:", np.max(g))
    print("x,y:", xy)
    print("getxy:", g[xy[1], xy[0]])

    copy_g = filtrar_g(g, g[xy[1], xy[0]], False)
    get_direction(copy_g)

    print("Finished...!")

