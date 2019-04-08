# Este script foi criado para fazer a retificação de uma região retangular
# sobre o plano d'água. Ele deverá ser excluído após seu conteúdo ser mesclado
# com o projeto principal, possivelmente com o módulo src.utils.utils.
import sys
# sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2, exifread, numpy as np, matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Polygon
from src.utils.utils import distance_euclidean, get_vanishLine_automatic
from ..utils.utils import *
from ..utils.wavelength import binarized_dilate_image

GROUND_HEIGHT = 750  # From sea level, in centimeters (source: Google Maps).
FIFTH_FLOOR_HEIGHT = 375 * 4  # From ground, in centimeters (source: building plan).
TRIPOD_HEIGHT = 140  # From floor, in centimeters.
CAMERA_HEIGHT = GROUND_HEIGHT + FIFTH_FLOOR_HEIGHT + TRIPOD_HEIGHT  # The camera height from the mean water surface plane, in centimeters.

CMOS_SENSOR_WIDTH = 23.5 / 10  # In centimeters (source: Nikon D3300 manual).
CMOS_SENSOR_HEIGHT = 15.6 / 10  # In centimeters (source: Nikon D3300 manual).
CMOS_SKEW = 0.0  # This is the expected value.

ROI_SIZE = (9000 + 400, 9000 + 9000)  # The size of the ROI, in centimeters.
PIXEL_SIZE = (10, 10)  # The size of pixels in the warped image, in centimeters.

CV2_WINDOW_RESIZE_WIDTH = 1200  # resize image to show on screen with openCV
CV2_WINDOW_RESIZE_HEIGHT = 800  # resize image to show on screen with openCV


def get_roi_lower_left_corner(im: np.ndarray) -> np.ndarray:
    """Return the location of the lower-left corner of the ROI in image coordinates.
    :param im: the input image.
    :return: the coordinates of the lower-left corner of the ROI.
    """
    coords = None

    def mouse_down_callback(event, x, y, flags, param):
        nonlocal coords
        if event == cv2.EVENT_LBUTTONDOWN:
            coords = np.asarray((x, y))

    cv2.namedWindow('roi-lower-left-corner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('roi-lower-left-corner', CV2_WINDOW_RESIZE_WIDTH, CV2_WINDOW_RESIZE_HEIGHT)
    cv2.setMouseCallback('roi-lower-left-corner', mouse_down_callback)
    while coords is None:
        cv2.imshow('roi-lower-left-corner', im)
        cv2.waitKey(33)
    cv2.destroyWindow('roi-lower-left-corner')
    
    return coords


def exif_ratio_to_float(value: exifread.utils.Ratio) -> float:
    """Convert a exifread.utils.Ratio value to float.
    :param value: a value from Exif metadata.
    :return: the value converted to float.
    """
    return float(value.num) / float(value.den)


def make_intrinsic_matrix(f: float, s: float, m: Tuple[float, float], o: Tuple[float, float]) -> np.ndarray:
    """Return the matrix with the intrinsic parameters of the camera. This matrix maps the coordinates from the camera coordinates system into the homogeneous image coordinates.
    :param f: the focal length, in centimeters.
    :param s: the skew of the pixel grid.
    :param m: (m_x, m_y) are the ratios between the sensor and the image measurements (width and height, in centimetes for the sensor and in pixels for the image).
    :param o: (o_u, o_v) is the location of the center of the image, in pixels.
    :return: the 3x3 matrix K.
    """
    return np.stack(((f * m[0], 0, 0), (s, -f * m[1], 0), (o[0], o[1], 1)), axis=1)


def make_rotation_matrix(K: np.ndarray, l_: np.ndarray) -> np.ndarray:
    """Return the rotation matrix that aligns the world coordinates system to the camera coordinates system.
    :param K: the 3x3 matrix with the intrinsic parameters of the camera.
    :param l_: the array with the coefficients of the vanishing line, in image coordinates.
    :return: the 3x3 matrix R.
    """
    normalize = lambda coords: coords / np.linalg.norm(coords)

    n = normalize(K.transpose().dot(l_))
    if n[1] < 0:
        n *= -1
    y = normalize(np.cross([1, 0, 0], n))
    x = np.cross(n, y)
    return np.stack((x, y, n), axis=1)


def make_translation_matrix(c: np.ndarray) -> np.ndarray:
    """Return the translation matrix responsible for centering the world on the center of the camera.
    :param c: the array with the location of the center of the camera in world coordinates system.
    :return: the 3x4 matrix T.
    """
    return np.stack(((1, 0, 0), (0, 1, 0), (0, 0, 1), -c), axis=1)


def compute_roi(im: np.ndarray, h: float, f: float, s: float, m: Tuple[float, float], o: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Region of Interest (ROI) after warping.
    :param im: the input image.
    :param h: the camera height from the mean water surface plane, in centimeters.
    :param f: the focal length, in centimeters.
    :param s: the skew of the pixel grid.
    :param m: (m_x, m_y) are the ratios between the sensor and the image measurements (width and height, in centimetes for the sensor and in pixels for the image).
    :param o: (o_u, o_v) is the location of the center of the image, in pixels.
    :return: (roi, q_, l_) includes the warpped ROI image, the coordinates of the ROI in the input image, and the coefficients of the vanishing line in the input image, respectively.
    """
    c = np.asarray((0, 0, h))  # The location of the center of the camera in world coordinates system.
    l_ = get_vanishLine_automatic()  # The coefficients of the vanishing line.
    q0_ = get_roi_lower_left_corner(im)  # The location of the lower-left corner of the ROI in image coordinates.

    # Compute the projection matrix P.
    K = make_intrinsic_matrix(f, s, m, o)
    R = make_rotation_matrix(K, l_)
    M = K.dot(R)
    T = make_translation_matrix(c)
    P = M.dot(T)  # The 3x4 projection matrix P maps points in 3D world coordinates to points in 2D image coordinates.

    # Compute the location of the reference point of the ROI in world coordinates (it is given by the intersecion between the ray r0 and the mean water plane).
    d0 = np.linalg.inv(M).dot((q0_[0] + 0.5, q0_[1] + 0.5, 1))  # The direction of the ray to the reference point in world coordinates.
    q0 = (-c[2] / d0[2]) * d0 + c  # The location of the reference point of the ROI in world coordinates.
    print("Print d0 y q0:", d0, q0)
    # Compute the location of the four corners of the ROI in world coordinates.
    q = [q0, q0 + np.asarray((ROI_SIZE[0], 0, 0)), q0 + np.asarray((ROI_SIZE[0], ROI_SIZE[1], 0)), q0 + np.asarray((0, ROI_SIZE[1], 0))]

    # Compute the location of the four corners of the ROI in image coordinates.
    q_ = [P.dot((x, y, z, 1)) for x, y, z in q]
    q_ = np.float32([np.asarray((x_ / w_, y_ / w_)) for x_, y_, w_ in q_])
    print("Print q y q_:", q, q_)
    new_h = distance_euclidean(q_[0], q_[1])
    new_w = distance_euclidean(q_[0], q_[3])
    w_ = np.float32([(0, 0), (ROI_SIZE[1] // PIXEL_SIZE[0], 0), (ROI_SIZE[1] // PIXEL_SIZE[0], ROI_SIZE[0] // PIXEL_SIZE[1]), (0, ROI_SIZE[0] // PIXEL_SIZE[1])])
    #w_ = np.float32([(0, 0), (int(new_h // PIXEL_SIZE[0]), 0), (int(new_h // PIXEL_SIZE[0]), int(new_w // PIXEL_SIZE[1])), (0, int(new_w // PIXEL_SIZE[1]))])
    #w_ = np.float32([(0, 0), (new_h, 0), (new_h, new_w), (0, new_w)])

    M = cv2.getPerspectiveTransform(q_, w_)
    roi = cv2.warpPerspective(im, M, (ROI_SIZE[1] // PIXEL_SIZE[0], ROI_SIZE[0] // PIXEL_SIZE[1]), flags=cv2.INTER_LANCZOS4)
    #roi = cv2.warpPerspective(im, M, (int(new_h // PIXEL_SIZE[0]), int(new_w // PIXEL_SIZE[1])), flags=cv2.INTER_LANCZOS4)
    #roi = cv2.warpPerspective(im, M, (new_h, new_w), flags=cv2.INTER_LANCZOS4)

    return roi, q_, l_


def find_skeleton3(img):
    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)

    _,thresh = cv2.threshold(img,127,255,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton,iters)

def compute_roi_v2(im: np.ndarray, h: float, f: float, s: float, m: Tuple[float, float], o: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Region of Interest (ROI) after warping.
    :param im: the input image.
    :param h: the camera height from the mean water surface plane, in centimeters.
    :param f: the focal length, in centimeters.
    :param s: the skew of the pixel grid.
    :param m: (m_x, m_y) are the ratios between the sensor and the image measurements (width and height, in centimetes for the sensor and in pixels for the image).
    :param o: (o_u, o_v) is the location of the center of the image, in pixels.
    :return: (roi, q_, l_) includes the warpped ROI image, the coordinates of the ROI in the input image, and the coefficients of the vanishing line in the input image, respectively.
    """
    c = np.asarray((0, 0, h))  # The location of the center of the camera in world coordinates system.
    l_ = get_vanishLine_automatic()  # The coefficients of the vanishing line.
    q0_ = get_roi_lower_left_corner(im)  # The location of the lower-left corner of the ROI in image coordinates.

    # Compute the projection matrix P.
    K = make_intrinsic_matrix(f, s, m, o)
    R = make_rotation_matrix(K, l_)
    M = K.dot(R)
    T = make_translation_matrix(c)
    P = M.dot(T)  # The 3x4 projection matrix P maps points in 3D world coordinates to points in 2D image coordinates.

    # Compute the location of the reference point of the ROI in world coordinates (it is given by the intersecion between the ray r0 and the mean water plane).
    d0 = np.linalg.inv(M).dot((q0_[0] + 0.5, q0_[1] + 0.5, 1))  # The direction of the ray to the reference point in world coordinates.
    q0 = (-c[2] / d0[2]) * d0 + c  # The location of the reference point of the ROI in world coordinates.

    # Compute the location of the four corners of the ROI in world coordinates.
    q = [q0, q0 + np.asarray((ROI_SIZE[0], 0, 0)), q0 + np.asarray((ROI_SIZE[0], ROI_SIZE[1], 0)), q0 + np.asarray((0, ROI_SIZE[1], 0))]

    # Compute the location of the four corners of the ROI in image coordinates.
    q_ = [P.dot((x, y, z, 1)) for x, y, z in q]
    q_ = np.float32([np.asarray((x_ / w_, y_ / w_)) for x_, y_, w_ in q_])

    q_w = int(distance_euclidean(q_[0], q_[1]))
    q_h = int(distance_euclidean(q_[0], q_[3]))
    y0 = int(q_[3][1])
    x0 = int(q_[0][0])

    # draw points
    draw_points = False
    if draw_points:
        cv2.circle(im, (q_[0][0], q_[0][1]), 23, (0, 0, 255), -1)       # red
        cv2.circle(im, (q_[1][0], q_[1][1]), 23, (0, 255, 0), -1)       # Green
        cv2.circle(im, (q_[2][0], q_[2][1]), 23, (255, 0, 0), -1)      # Blue
        cv2.circle(im, (q_[3][0], q_[3][1]), 23, (0, 255, 255), -1)     # Yellow
        #cv2.circle(im, (x0, y0), 23, (255, 0, 255), -1)     # Purpura

    crop_img = im[y0:y0 + q_h, x0:x0 + q_w]
    crop_img = binarized_dilate_image(crop_img, show_=False, bin_cv2=True)
    skeleton, iters = find_skeleton3(crop_img)
    print("Iters skeleton: ", iters)

    #show_image(im, "ROI-X", CV2_WINDOW_RESIZE_WIDTH, CV2_WINDOW_RESIZE_HEIGHT)
    show_image(crop_img, "ROI-X", q_w, q_h)
    show_image(skeleton, "ROI-skeleton", q_w, q_h)

    return []

def read_exif_tags(path_name: str) -> dict:
    """Return the Exif metadata from the given TIFF or JPEG filename.
    :param path_name: the path to the image file.
    :return: a dictionary with the extracted tags.
    """
    with open(path_name, 'rb') as f:
        return exifread.process_file(f)


def main():
    # Set some constant values.
    image_path = 'src/images/image_test/'
    image_filename = 'DSC_000000920'

    # Get intrinsic parameters of the camera from the input image file.
    tags = read_exif_tags(image_path + image_filename + '.jpg')

    focal_length = exif_ratio_to_float(tags['EXIF FocalLength'].values[0])  # In millimeters.
    image_width = tags['EXIF ExifImageWidth'].values[0]  # In pixels.
    image_height = tags['EXIF ExifImageLength'].values[0]  # In pixels.

    f = focal_length / 10  # The focal length, in centimeters.
    s = CMOS_SKEW  # The skew of the pixel grid.
    m = (image_width / CMOS_SENSOR_WIDTH, image_height / CMOS_SENSOR_HEIGHT)  # (m_x, m_y) are the ratios between the sensor and the image measurements (width and height, in centimetes for the sensor and in pixels for the image).
    o = (image_width / 2, image_height / 2)  # (o_u, o_v) is the location of the center of the image, in pixels.

    # Load input image.
    im = cv2.imread(image_path + image_filename + '.jpg')  # The input image.

    # Compute the image of the ROI.
    roi, q_, l_ = compute_roi(im, CAMERA_HEIGHT, f, s, m, o)

    # Show results.
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(im[..., ::-1])
    ax1.set(title='Input')
    ax1.add_patch(Polygon(q_, edgecolor='y', fill=None))

    x = np.array(ax1.get_xlim())
    y = -(l_[0] * x + l_[2]) / l_[1]
    ax1.add_line(Line2D(x, y, color='r'))
    plt.axis('off')
    plt.savefig(image_path + image_filename + "_show_roi.jpg", bbox_inches='tight')

    roi = cv2.flip(roi, 0)  # horizontal flip image
    # ax2.imshow(roi[..., ::-1])
    # ax2.set(title='ROI')
    plt.imsave(image_path + image_filename + "_crop.jpg", roi[..., ::-1])
    # plt.show(block=True)


if __name__ == '__main__':
    main()