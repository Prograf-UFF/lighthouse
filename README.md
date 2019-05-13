# COMPUTING VESSEL VELOCITY FROM SINGLE PERSPECTIVE PROJECTION IMAGES
We present an image-based approach to estimate the velocity of moving vessels from their traces on the water surface. Vessels moving at constant heading and speed display a familiar V-shaped pattern which only differs from one to another by the wavelength of their transverse and divergent components. Such wavelength is related to vessel velocity. We use planar homography and natural constraints on the geometry of ships' wake crests to compute vessel velocity from single optical images acquired by conventional cameras.

The approach was developed by Jose Luis Huillca and [Leandro A. F. Fernandes](http://www.ic.uff.br/~laffernandes).

This repository includes the implementation of the algorithm, and a sample application using this implementation.

Please cite our IEEE ICIP 2019 paper if you use this code in your research:
```
@InProceedings{huillca_fernandes-icip-2019,
  author    = {Huillca, Jose Luis and Fernandes, Leandro A. F.},
  title     = {Computing vessel velocity from single perspective projection images},
  booktitle = {Proceedings of the 2019 IEEE International Conference on Image Processing (ICIP)},
  year      = {2019},
  pages     = {(to appear)}
}
```

## Licence
All code is released under the [GNU General Public License](https://www.gnu.org/licenses/), version 3, or (at your option) any later version.

## Platforms
We have compiled and tested the sample application on Linux and Windows using Python 3.6.

## Requirements
Make sure that you have all the following tools and libraries installed and working before attempting to compile.

Required tools:
- [Python](https://pt.wikipedia.org/wiki/Python) 3.6 or later (Linux or Windows)
- [OpenCV](https://pt.wikipedia.org/wiki/OpenCV) 3 or later (Linux or Windows)

Required Python packages:
- [ExifRead](https://pypi.org/project/ExifRead/) 2.1.2 or later 
- [SciPy](https://www.scipy.org/about.html) 0.17 or later
- [Scikit-learn](https://scikit-learn.org/) 0.15.2 or later
- [NumPy](https://www.numpy.org/) 1.11 or later
- [Matplotlib](https://matplotlib.org/) 3.0.3 or later

## Building, Compiling, and Running
Use the [git clone](https://git-scm.com/docs/git-clone) command to download the project:
```bash
$ git clone https://github.com/Prograf-UFF/lighthouse.git
$ cd lighthouse
````

The images used for the test are in the repository, both the images in **RGB** and the edges's images. [RCF](https://github.com/yun-liu/rcf) (***Richer Convolutional Features***) was used for edge detection.

## Demo
```
sh test.sh
```
