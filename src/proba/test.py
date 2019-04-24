from src.utils.sandbox import compute_roi_v2, read_exif_tags, exif_ratio_to_float, CMOS_SENSOR_HEIGHT, CMOS_SENSOR_WIDTH, CMOS_SKEW, CAMERA_HEIGHT
from src.utils.utils import show_image
import cv2


def test_func(path: str, image_name: str):
    # Get intrinsic parameters of the camera from the input image file.
    tags = read_exif_tags(path + image_name + '.jpg')
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
    im = cv2.imread(path + image_name + '_edge.png')  # The input image.
    # Image 'png' with BGRA color, and we need only BGR
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)

    # capturamos el ROI en escala de grises
    crop_img, _, _ = compute_roi_v2(im, CAMERA_HEIGHT, f, s, m, o)
    # binarizamos la imagen con otsu
    _, image_bin = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show_image(crop_img, "CROP IMAGE", crop_img.shape[1], crop_img.shape[0])
    show_image(image_bin, "BINARY IMAGE", crop_img.shape[1], crop_img.shape[0])

    # aplicar un algoritmo de thining a la imagen binaria
    
