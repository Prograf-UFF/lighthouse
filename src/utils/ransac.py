from sklearn import linear_model
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np


def ransac_base(sample_x: np.ndarray, sample_y: np.ndarray, show_: bool=False, figure_num: int=3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Linear model estimation
    :param sample_x: coordinates of the 'x' axis of the sample to fit
    :param sample_y: coordinates of the 'y' axis of the sample to fit
    :param show_: if is true, plot inliers, outliers and RANSAC regressor
    :param figure_num: if show_ is true, is a number of figure to plot
    :return:
    """
    x, y = sample_x, sample_y

    # Robustly fit linear model with RANSAC algorithm
    ransac_model = linear_model.RANSACRegressor()
    ransac_model.fit(x, y)
    inlier_mask = ransac_model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_x = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y_ransac = ransac_model.predict(line_x)

    if show_:
        plt.figure(figure_num)

        lw = 2  # line weight
        plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
        plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
        plt.plot(line_x, line_y_ransac, color='cornflowerblue', linewidth=lw, label='RANSAC regressor')
        plt.legend(loc='lower right')
        plt.xlabel("Input")
        plt.ylabel("Response")
        plt.show()

    return line_x, line_y_ransac, inlier_mask
