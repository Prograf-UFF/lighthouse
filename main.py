from src.rectifyAffine.rectifyAffine import RectifyAffine
import numpy as np
from src.utils.utils import *
from src.utils.ransac import *
from src.utils.canny import *
from src.utils.otsu import *
from src.utils.sobel import *


if __name__ == "__main__":
    path = 'src/images/'
    img = 'exemplo2.jpg'

    rectAffine = RectifyAffine(path=path, image=img)
    img_result = rectAffine.image_rectification(show_=True)
    rectAffine.get_lineV(img_result, auto=False, sobel_show=False, show_filtrar_g=False, show_ransac=False)

    print("Finished...!")

