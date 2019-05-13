import matplotlib.pyplot as plt
from ..utils.utils import show_image, WAVE_WINDOW_SIZE_HEIGHT, WAVE_WINDOW_SIZE_WIDTH
from ..utils.wavelength import get_wave_edge, smooth_wave, find_max_locals, get_wavelength, get_estimated_speed, ms_to_nudo, binarized_dilate_image
from ..utils.sandbox import read_exif_tags, exif_ratio_to_float, CAMERA_HEIGHT, CMOS_SKEW, CMOS_SENSOR_WIDTH, CMOS_SENSOR_HEIGHT, compute_roi
import cv2, numpy as np


class RectifyAffine:
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
        roi, q_, l_ = compute_roi(im, CAMERA_HEIGHT, f, s, m, o)
        roi = cv2.flip(roi, 0)  # horizontal flip image
        if show_:
            # Show results.
            show_image(roi, "ROI", WAVE_WINDOW_SIZE_WIDTH, WAVE_WINDOW_SIZE_HEIGHT)
        return roi[..., ::-1]

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
