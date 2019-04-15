import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
import matplotlib.pyplot as plt
from ..utils.utils import *
from ..utils.ransac import *
from ..utils.canny import *
from ..utils.sobel import sobel_filter, sort_g, cdf, get_xy, filtrar_g
from ..utils.wavelength import get_wave_edge, smooth_wave, find_max_locals, get_wavelength, get_estimated_speed, ms_to_nudo, binarized_dilate_image
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from ..utils.sandbox import read_exif_tags, exif_ratio_to_float, CAMERA_HEIGHT, CMOS_SKEW, CMOS_SENSOR_WIDTH, CMOS_SENSOR_HEIGHT, compute_roi, compute_roi_v2
import cv2, exifread, copy, random


class RectifyAffine:
    # ---------- Settings ---------
    nLinePairs = 2  # Select this may pairs of perpendicular lines

    def __init__(self, path, image_name):
        self.path = path
        self.image_name = image_name
        self.my_image = []

    def image_rectification(self, show_: bool=False) -> np.ndarray:
        """ We get the rectified image
        :param show_: if is true, plot result
        :return: image, a part of the rectified image
        """
        # Get intrinsic parameters of the camera from the input image file.
        tags = read_exif_tags(self.path + self.image_name + '.jpg')
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
        im = cv2.imread(self.path + self.image_name + '_edge.png')  # The input image.
        # Image 'png' with BGRA color, and we need only BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

        # Compute the image of the ROI.
        crop_img = compute_roi_v2(im, CAMERA_HEIGHT, f, s, m, o)
        # binary image put into gray-scale image
        a = np.zeros((crop_img.shape[0], crop_img.shape[1], 3), dtype=im.dtype)
        a[:, :, 0] = crop_img
        a[:, :, 1] = crop_img
        a[:, :, 2] = crop_img

        #  We obtain the coordinates represented by the wave
        #wave_coord = get_wave_edge(crop_img, show_=False)
        #get_direction(wave_coord, [], get_q_auto=True, show_ransac=False, show_result=True)
        self.get_lineV(a, True, True, True, True)


        return []
        # roi, q_, l_ = compute_roi(im, CAMERA_HEIGHT, f, s, m, o)
        roi = cv2.flip(roi, 0)  # horizontal flip image
        if show_:
            # Show results.
            show_image(roi, "ROI", WAVE_WINDOW_SIZE_WIDTH, WAVE_WINDOW_SIZE_HEIGHT)
        return roi[..., ::-1]

    # function not necessary
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
        #processed_image = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
        processed_image = cv2.bilateralFilter(image, 11, 17, 17)
        g = sobel_filter(processed_image, 3, sobel_show)
        sort_g_tupla = sort_g(g)
        acu_cdf = cdf(sort_g_tupla, False)

        rd = random.uniform(0.82, 0.86)
        xy = get_xy(rd, acu_cdf, sort_g_tupla)
        print("random #:", rd)

        copy_g = filtrar_g(g, g[xy[1], xy[0]], show_filtrar_g)
        get_direction(copy_g, img, get_q_auto=auto, show_ransac=show_ransac, show_result=True)

    def estimated_speed(self, image: np.ndarray):
        """
        :param image: ROI's result in RGB
        :return: speed of the boat in knots
        """
        # binarize the image and apply dilation so that the curves are not very thick
        image_dilate = binarized_dilate_image(image, show_=True, bin_cv2=True)

        #  We obtain the coordinates represented by the wave
        wave_coord = get_wave_edge(image_dilate, show_=False)

        # we use the 'savgol_filter' algorithm to smooth the wave
        yhat = smooth_wave(wave_coord, show_=True)

        # Actually they were the local minimums, but as opencv (starting from 0 in the upper left corner)
        #  it shows the image differently from 'pyplot' (starting from 0 in the lower left corner)
        peaks_coords = find_max_locals(wave_coord, yhat, show_=True)
        wavelength = get_wavelength(peaks_coords)
        # we divide the wave length by 10 (pixel size in centimeters) to pass its distance in meters
        speed = get_estimated_speed(wavelength / 10.0)
        speed_knot = ms_to_nudo(speed)
        print("SPEED:", speed, " m/s")
        print("SPEED:", speed_knot, " knot")
        plt.show()
        return speed_knot
