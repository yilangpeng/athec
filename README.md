# Athec
A Python package for computational aesthetic analysis of visual media

Athec is a Python library that measures a variety of aesthetic attributes, such as brightness, contrast, colorfulness, color variety, percentages of different colors, visual complexity, and depth of field. Computationally calculated visual attributes have been demonstrated to predict a wide range of outcomes, such as images' aesthetic appeal, popularity on social media, and interestingness. 

## How to use
1. Install the following packages before running the scripts
numpy, Pillow, matplotlib, OpenCV, Scipy, scikit-image, pyemd

The current version has been tested on the folllowing versions:
* Python: 3.9
* numpy: 1.20.3
* Pillow: 8.2.0
* matplotlib: 3.4.2
* opencv-contrib-python: 4.5.2.54
* scipy: 1.6.3
* scikit-image: 0.18.1
* pyemd: 0.5.1

2. Run the demo scripts. The documentation about each function is also provided in these scripts.
Note: please directly download the scripts from this folder instead of using pip in termal.
After that, make sure you add the package path to the sys paths (see example below):

import os, sys
athec_path = os.path.expanduser("~/Documents/Workspace/Computer vision/Athec/")
sys.path.append(athec_path)

## Citation
```
@article{peng2021athec,
  title={Athec: A Python Library for Computational Aesthetic Analysis of Visual Media in Social Science Research},
  author={Peng, Yilang},
  journal={Computational Communication Research},
  year={Forthcoming}
}

@article{peng2018feast,
  title={Feast for the Eyes: Effects of Food Perceptions and Computer Vision Features on Food Photo Popularity.},
  author={Peng, Yilang and Jemmott III, John B},
  journal={International Journal of Communication},
  volume={12},
  year={2018}
}

```

