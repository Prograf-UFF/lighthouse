from src.rectifyAffine.rectifyAffine import RectifyAffine
import numpy as np


if __name__ == "__main__":
    path = 'src/images/'
    img = 'exemplo1.jpg'

    rectAffine = RectifyAffine(path=path, image=img)
    rectAffine.get_perspective_transform()
    # print(np.cross([1, 3, 1], [2, 4, 1]))
    print("Finished...!")

