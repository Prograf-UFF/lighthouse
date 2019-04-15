import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np

from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def ransac_base(sample_x: np.ndarray, sample_y: np.ndarray, show_: bool=False, figure_num: int=3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Linear model estimation
    :param sample_x: coordinates of the 'x' axis of the sample to fit
    :param sample_y: coordinates of the 'y' axis of the sample to fit
    :param show_: if is true, plot inliers, outliers and RANSAC regressor
    :param figure_num: if show_ is true, is a number of figure to plot
    :return:
    """
    x, y = sample_x, sample_y

    '''np.random.seed(42)
    x = np.random.normal(size=400)
    y = np.sin(x)
    # Make sure that it X is 2D
    x = x[:, np.newaxis]'''

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


def ransac(X: np.ndarray, y: np.ndarray, show_: bool=False, figure_num: int=3):
    '''np.random.seed(42)
    X = np.random.normal(size=400)
    y = np.sin(X)'''
    # Make sure that it X is 2D
    X = X[:, np.newaxis]

    estimators = [('RANSAC', RANSACRegressor(random_state=42))]
    colors = {'RANSAC': 'lightgreen'}
    linestyle = {'RANSAC': '--'}
    lw = 3

    x_plot = np.linspace(X.min(), X.max())
    for title, this_X, this_y in [
        ('Modeling Errors Only', X, y)]:
        plt.figure(figsize=(1, 1))
        plt.plot(this_X[:, 0], this_y, 'b+')

        for name, estimator in estimators:
            model = make_pipeline(PolynomialFeatures(3), estimator)
            model.fit(this_X, this_y)
            y_plot = model.predict(x_plot[:, np.newaxis])

            plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                     linewidth=lw, label='%s' % name)

        plt.legend(loc='upper right', frameon=False, title="")
        plt.title(title)
    plt.show()
    return x_plot, [], []


if __name__ == "__main__":
    print("Robust linear estimator fitting")
    # robustEstimators()
    ransac([],[], True)
