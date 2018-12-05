import sys
import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
import matplotlib.pyplot as plt
from ..utils.utils import *
from ..utils.ransac import *
from ..utils.canny import *
from ..utils.sobel import *
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from ..utils.sandbox import read_exif_tags, exif_ratio_to_float, CAMERA_HEIGHT, CMOS_SKEW, CMOS_SENSOR_WIDTH, CMOS_SENSOR_HEIGHT, compute_roi
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2, exifread, copy, random


class RectifyAffine:
    # ---------- Settings ---------
    nLinePairs = 2  # Select this may pairs of perpendicular lines

    def __init__(self, path, image):
        self.path = path
        self.image = image
        self.my_image = []

    def image_rectification(self, show_: bool=False) -> np.ndarray:
        """ We get the rectified image
        :param show_: if is true, plot result
        :return: image, a part of the rectified image
        """
        # Get intrinsic parameters of the camera from the input image file.
        tags = read_exif_tags(self.path + self.image)
        focal_length = exif_ratio_to_float(tags['EXIF FocalLength'].values[0])  # In millimeters.
        image_width = tags['EXIF ExifImageWidth'].values[0]  # In pixels.
        image_height = tags['EXIF ExifImageLength'].values[0]  # In pixels.

        f = focal_length / 10  # The focal length, in centimeters.
        s = CMOS_SKEW  # The skew of the pixel grid.
        # (m_x, m_y) are the ratios between the sensor and the image measurements
        # (width and height, in centimetes for the sensor and in pixels for the image).
        m = (image_width / CMOS_SENSOR_WIDTH, image_height / CMOS_SENSOR_HEIGHT)
        o = (image_width / 2, image_height / 2)  # (o_u, o_v) is the location of the center of the image, in pixels.

        # Load input image.
        im = cv2.imread(self.path + self.image)  # The input image.

        # Compute the image of the ROI.
        roi, q_, l_ = compute_roi(im, CAMERA_HEIGHT, f, s, m, o)
        roi = cv2.flip(roi, 0)  # horizontal flip image
        if show_:
            # Show results.
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            ax1.imshow(im[..., ::-1])
            ax1.set(title='Input')
            ax1.add_patch(Polygon(q_, edgecolor='y', fill=None))

            x = np.array(ax1.get_xlim())
            y = -(l_[0] * x + l_[2]) / l_[1]
            ax1.add_line(Line2D(x, y, color='r'))

            ax2.imshow(roi[..., ::-1])
            ax2.set(title='ROI')
            plt.show(block=True)
        return roi[..., ::-1]

    def get_lineV(self, img: np.ndarray, auto: bool=True, sobel_show: bool=False, show_filtrar_g: bool=False, show_ransac: bool=False) -> None:
        """ We get the trace lines left by the boat
        :param img: rectified image
        :param auto: if true, select the area of the trail left by the boat automatically
        :param sobel_show: plot sobel result
        :param show_filtrar_g: plot filtrar_g function result
        :param show_ransac: plot RANSAC result
        :return: None
        """
        image = copy.copy(img)
        # https://docs.opencv.org/3.0-beta/modules/photo/doc/denoising.html
        # we use a filter to eliminate noise
        processed_image = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        g = sobel_filter(processed_image, 3, sobel_show)
        sort_g_tupla = sort_g(g)
        acu_cdf = cdf(sort_g_tupla, False)

        rd = random.uniform(0.82, 0.86)
        xy = get_xy(rd, acu_cdf, sort_g_tupla)
        print("random #:", rd)

        copy_g = filtrar_g(g, g[xy[1], xy[0]], show_filtrar_g)
        get_direction(copy_g, img, get_q_auto=auto, show_ransac=show_ransac, show_result=True)

