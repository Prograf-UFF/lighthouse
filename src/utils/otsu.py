import sys
import matplotlib.pyplot as plt
import math
from PIL import Image
import numpy as np
# to work openCV on python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


# -------------------------OTSU-----------------
threshold_values = {}


def Hist(img, show_=False):
    row, col = img.shape
    y = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            y[img[i, j]] += 1
    if show_:
        x = np.arange(0, 256)
        plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
        plt.show()
    return y


def regenerate_img(img, threshold):
    row, col = img.shape
    y = np.zeros((row, col))
    for i in range(0, row):
        for j in range(0, col):
            if img[i, j] >= threshold:
                y[i, j] = 255
            else:
                y[i, j] = 0
    return y


def countPixel(h_):
    cnt = 0
    for i in range(0, len(h_)):
        if h_[i] > 0:
            cnt += h_[i]
    return cnt


def wieght(s, e, h_):
    w = 0
    for i in range(s, e):
        w += h_[i]
    return w


def mean(s, e, h_):
    m = 0
    w = wieght(s, e, h_)
    for i in range(s, e):
        m += h_[i] * i

    return m / float(w)


def variance(s, e, h_):
    v = 0
    m = mean(s, e, h_)
    w = wieght(s, e, h_)
    for i in range(s, e):
        v += ((i - m) ** 2) * h_[i]
    v /= w
    return v


def threshold(h_):
    cnt = countPixel(h_)
    for i in range(1, len(h_)-1):
        vb = variance(0, i, h_)
        wb = wieght(0, i, h_) / float(cnt)
        mb = mean(0, i, h_)

        vf = variance(i, len(h_), h_)
        wf = wieght(i, len(h_), h_) / float(cnt)
        mf = mean(i, len(h_), h_)

        V2w = wb * (vb) + wf * (vf)
        V2b = wb * wf * (mb - mf) ** 2

        fw = open("trace.txt", "a")
        fw.write('T=' + str(i) + "\n")

        fw.write('Wb=' + str(wb) + "\n")
        fw.write('Mb=' + str(mb) + "\n")
        fw.write('Vb=' + str(vb) + "\n")

        fw.write('Wf=' + str(wf) + "\n")
        fw.write('Mf=' + str(mf) + "\n")
        fw.write('Vf=' + str(vf) + "\n")

        fw.write('within class variance=' + str(V2w) + "\n")
        fw.write('between class variance=' + str(V2b) + "\n")
        fw.write("\n")

        if not math.isnan(V2w):
            threshold_values[i] = V2w


def get_optimal_threshold():
    min_V2w = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    print('optimal threshold: ', optimal_threshold[0])
    return optimal_threshold[0]


def otsu(path, img, show_=False):
    image = Image.open(path + "/" + img).convert("L")
    img = np.asarray(image)

    h = Hist(img)
    threshold(h)
    op_thres = get_optimal_threshold()

    res = regenerate_img(img, op_thres)
    if show_:
        plt.imshow(res, cmap='gray')
        plt.show()
    return res