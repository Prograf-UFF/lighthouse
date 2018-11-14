import sys
import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
import matplotlib.pyplot as plt
from ..utils.utils import *
from ..utils.ransac import *
from ..utils.canny import *
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2

class RectifyAffine:
    # ---------- Settings ---------
    nLinePairs = 2  # Select this may pairs of perpendicular lines

    def __init__(self, path, image):
        self.path = path
        self.image = image
        self.image_save = path + "save_images/" + image

    @staticmethod
    def system_of_equations(c_, d_, e_, limit_x_, limit_y_):
        A = [[], [], [], [], [], []]
        A[0] = [c_[0], c_[1], 1, 0, 0, 0, 0]
        A[1] = [0, 0, 0, c_[0], c_[1], 1, 0]
        A[2] = [d_[0], d_[1], 1, 0, 0, 0, -(limit_x_ - 1)]
        A[3] = [0, 0, 0, d_[0], d_[1], 1, 0]
        A[4] = [e_[0], e_[1], 1, 0, 0, 0, 0]
        A[5] = [0, 0, 0, e_[0], e_[1], 1, -(limit_y_ - 1)]
        return A

    @staticmethod
    def translation_matrix(vh):
        T = np.identity(3)
        T[0][0] = vh[6][0]
        T[0][1] = vh[6][1]
        T[0][2] = vh[6][2]
        T[1][0] = vh[6][3]
        T[1][1] = vh[6][4]
        T[1][2] = vh[6][5]
        T[2][2] = vh[6][6]
        return T

    def test_sanidad(self, M, m_origin, m_result, a_legenda):
        m_result_ = [np.dot(M, x) for x in m_result]
        result = []
        for i in range(0, len(a_legenda)):
            x = m_result_[i]
            o = [a_legenda[i] + "_o"]
            r = [a_legenda[i] + "_r"]
            for j in range(0, 3):
                r.append(x[j]/x[2])
                o.append(m_origin[i][j])
            result.append(o)
            result.append(r)
        #print(result)
        my_print(["coord", "a", "b", "c"], np.array(result), "Test Sanidade")


    def matrix_h(self):
        im = image_read(self.path, self.image)
        show_image_properties(im)
        plt.close()  # ginput does not allow new points to be plotted
        imshow(im, cmap='gray')
        axis('image')

        # Create figure
        plt.figure(1)
        #vLine = get_vanishLine_manual(im, False, False)
        vLine = get_vanishLine_automatic(im)
        #vLine = get_vanish_line(im.shape)
        #show()
        #return
        # Create a Homographic matrix
        H = np.identity(3)
        H[2, 0] = vLine[0]
        H[2, 1] = vLine[1]
        H[2, 2] = vLine[2]

        my_print([], H, "H")
        # Compute transformed line = H^-T * l
        #H_ = np.linalg.inv(H)
        #print_array("H_T", H_)

        c = get_point()
        d = get_point()
        e = get_point()
        plt.suptitle('wait...')

        c_ = np.dot(H, c)
        d_ = np.dot(H, d)
        e_ = np.dot(H, e)
        c_ = [x/c_[2] for x in c_]
        d_ = [x/d_[2] for x in d_]
        e_ = [x/e_[2] for x in e_]

        # Limites de la nueva imagen
        limit_x_ = int(distance_euclidean(c, d))
        limit_y_ = int(distance_euclidean(c, e))

        A = self.system_of_equations(c_, d_, e_, limit_x_, limit_y_)
        my_print([], np.array(A), "A")

        _, _, vh = np.linalg.svd(A)
        my_print([], vh, "vh")

        T = self.translation_matrix(vh)
        my_print([], T, "T")

        self.test_sanidad(T,
                          [[0,0,1], [limit_x_-1,0,1], [0,limit_y_-1,1]],
                          [c_, d_, e_],
                          ["c", "d", "e"])

        M = np.dot(T, H)
        M_ = np.linalg.inv(M)

        self.test_sanidad(M_,
                          [c, d, e],
                          [[0, 0, 1], [limit_x_ - 1, 0, 1], [0, limit_y_ - 1, 1]],
                          ["res_c", "res_d", "res_e"])

        limit_x = im.shape[1]
        limit_y = im.shape[0]
        plt.figure(4)

        im_result = create_image([limit_y_,limit_x_], im.dtype, im.ndim)
        for x_ in range(0, limit_x_):
            for y_ in range(0, limit_y_):
                p = np.dot(M_, [x_, y_, 1])
                if p[2] != 0:
                    x = int(np.round(p[0]/p[2]))
                    y = int(np.round(p[1]/p[2]))
                    if (0 <= x < limit_x) and (0 <= y < limit_y):
                        im_result[y_, x_] = im[y, x]

        imshow(im_result, interpolation='nearest')
        show()
        plt.imsave(self.image_save, im_result)

