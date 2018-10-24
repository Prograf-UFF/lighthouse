from src.rectifyAffine.rectifyAffine import RectifyAffine
import numpy as np


if __name__ == "__main__":
    path = 'src/images/'
    img = 'exemplo1.jpg'

    rectAffine = RectifyAffine(path=path, image=img)
    #rectAffine.get_perspective_transform(1, 1, 1)
    rectAffine.matrix_h()

    print("Finished...!")

