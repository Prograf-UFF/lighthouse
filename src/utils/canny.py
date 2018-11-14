from src.utils.utils import *
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


def single_band_representation(img):
    img_r = create_image(img.shape, img.dtype, 3)
    w = img.shape[1]
    h = img.shape[0]
    color_titles = ('Banda Vermelha(Red)')
    for x in range(0, w):
        for y in range(0, h):
            cor = img[y, x]
            img_r[y, x] = [cor[0], cor[0], cor[0]]
    #imshow(img_r)
    #show()
    return img_r


# imagen a escala de grises
def get_canny(img_gray, limiar=45):
    lowThreshold = limiar  # 66/45
    max_lowThreshold = limiar*3  # 198/135
    kernel_size = 3
    # Reduce noise with a kernel 3x3
    detected_edges = cv2.blur(img_gray, (3, 3))

    # Canny detector
    detected_edges = cv2.Canny(detected_edges, lowThreshold, max_lowThreshold, kernel_size)
    return detected_edges


# -------------------------------------------
def empty(x):
    pass


def main_canny(path, image, l_limiar=0):
    src, src_gray = [], []
    dst, detected_edges = [], []

    edgeThresh = 1
    lowThreshold = l_limiar
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3
    window_name = "Edge Map"
    # ----------
    src = cv2.imread(path + image)
    if not src.data:
        return -1

    # src = single_band_representation(src)
    # cv2.imshow("Original", src)

    # Create a matrix of the same type and size as src(for dst)
    # dst = create_image(src.shape, src.dtype, 3)

    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Create a Trackbar for user to enter threshold
    cv2.createTrackbar("Min Threshold:", window_name, lowThreshold, max_lowThreshold, empty)
    #CannyThreshold()
    while True:
        # Reduce noise with a kernel 3x3
        detected_edges = cv2.blur(src_gray, (3, 3))

        # Canny detector
        detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, kernel_size)

        lowThreshold = cv2.getTrackbarPos("Min Threshold:", window_name)
        cv2.setTrackbarPos("Min Threshold:", window_name, lowThreshold)

        dst = detected_edges
        cv2.imshow(window_name, dst)
        if cv2.waitKey(1) == 27:  # 27 is the key 'esc'
            break
    cv2.destroyAllWindows()
    print("lowThreshold:", lowThreshold)
    print("lowThreshold * ratio:", lowThreshold * ratio)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def auto_canny_(path, image):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(path + image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)

    # show the images
    #cv2.imshow("Original", image)
    cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #imshow(tight, cmap="gray")
    #show()