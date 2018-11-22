import sys
import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
import matplotlib.pyplot as plt
from ..utils.utils import *
from ..utils.ransac import *
from ..utils.canny import *
from ..utils.sobel import *
import random
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
import copy

class RectifyAffine:
    # ---------- Settings ---------
    nLinePairs = 2  # Select this may pairs of perpendicular lines

    def __init__(self, path, image):
        self.path = path
        self.image = image
        self.my_image = []

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
            o = [a_legenda[i] + "_original"]
            r = [a_legenda[i] + "_resultante"]
            for j in range(0, 3):
                r.append(x[j]/x[2])
                o.append(m_origin[i][j])
            result.append(o)
            result.append(r)
        #print(result)
        my_print(["coord", "a", "b", "c"], np.array(result), "Test Sanidade")


    def matrix_h(self):
        im = image_read(self.path, self.image)
        self.my_image = copy.copy(im)
        show_image_properties(im)
        plt.close()  # ginput does not allow new points to be plotted
        imshow(im, cmap='gray')
        axis('image')

        # Create figure
        plt.figure(1)
        # vLine = get_vanishLine_manual(im, False, False)
        vLine = get_vanishLine_automatic(im)

        # Create a Homographic matrix
        H = np.identity(3)
        H[2, 0] = vLine[0]
        H[2, 1] = vLine[1]
        H[2, 2] = vLine[2]

        my_print([], H, "H")

        plt.suptitle('********* Clique tres pontos para pegar a imagem a ser corrigida *********')
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
                          ["rpt_c", "rpt_d", "rpt_e"])

        plt.close(1)
        show()
        return M_, limit_x_, limit_y_

    # Obtemos a imagem retificada passando a matriz M_, o tamanho
    # da imagem resultante é w = limite_x_ e h = limite_y_
    def image_rectification(self, M_, limit_x_, limit_y_):
        plt.figure(4)
        limit_x = self.my_image.shape[1]
        limit_y = self.my_image.shape[0]
        im_result = create_image([limit_y_, limit_x_], self.my_image.dtype, self.my_image.ndim)

        for x_ in range(0, limit_x_):
            for y_ in range(0, limit_y_):
                p = np.dot(M_, [x_, y_, 1])
                if p[2] != 0:
                    x = int(np.round(p[0] / p[2]))
                    y = int(np.round(p[1] / p[2]))
                    if (0 <= x < limit_x) and (0 <= y < limit_y):
                        im_result[y_, x_] = self.my_image[y, x]

        # imshow(im_result, interpolation='nearest')
        plt.close(4)
        show()
        # plt.imsave(self.path + 'img_save/' + self.image, im_result)
        return im_result


    # Obtemos as linhas de rastro deixadas pelo navio
    def get_lineV(self, img, auto=True, sobel_show=False, show_filtrar_g=False, show_ransac=False):
        image = copy.copy(img)
        # https://docs.opencv.org/3.0-beta/modules/photo/doc/denoising.html
        # nós usamos um filtro para eliminar o ruído
        processed_image = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        g = sobel_filter(processed_image, 3, sobel_show)
        sort_g_tupla = sort_g(g)
        acu_cdf = cdf(sort_g_tupla, False)

        rd = random.uniform(0.82, 0.86)
        xy = get_xy(rd, acu_cdf, sort_g_tupla)

        print("random #:", rd)
        print("g max val:", np.max(g))
        print("x,y:", xy)
        print("getxy:", g[xy[1], xy[0]])

        copy_g = filtrar_g(g, g[xy[1], xy[0]], show_filtrar_g)
        get_direction(copy_g, img, get_q_auto=auto, show_ransac=show_ransac, show_result=True)

