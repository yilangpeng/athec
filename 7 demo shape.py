import os
from athec import misc, edge, shape
import numpy as np

img_folder = os.path.join("image", "original")
resize_folder = os.path.join("image", "resize")
tf_folder = os.path.join("image", "transform")

imgname = "example1.jpg"
img = os.path.join(resize_folder, imgname)

'''
Perform Canny edge detection and results will be used in line detection.
'''
edges = edge.tf_edge_canny(img,
                           otsu_ratio = 0.5,
                           gaussian_blur_kernel = (5,5))

'''
Calculate line dynamics based on edge map.
Return:
(1) the number of lines detected
(2) summary statistics of line lengths.
(2) summary statistics of line orientations (including circular mean and circular standard deviation).
(1) the number of horizontal lines and summary statistics of their lengths.
(1) the number of vertical lines and summary statistics of their lengths.
(1) the number of slanting lines and summary statistics of their lengths.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
return_summary (optional, default False): bool. Whether summary statistics will be returned.
horizontal_degree (optional, default 10): int. Lines with orientations between -horizontal_degree and horizontal_degree will be categorized as horizontal lines.
vertical_degree (optional, default 80): int. Lines with orientations smaller than -vertical_degree or larger than vertical_degree will be categorized as vertical lines.
HoughLinesP_rho, HoughLinesP_theta, HoughLinesP_threshold, HoughLinesP_minLineLength, HoughLinesP_maxLineGap: parameters for line detection. The function uses Probabilistic Hough Line Transform to detect lines. See 
https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
'''
result = shape.attr_line_hough_edge(edges,
                              save_path = os.path.join(tf_folder, "line hough edge canny", imgname),
                              horizontal_degree = 10,
                              vertical_degree = 80,
                              HoughLinesP_rho = 1,
                              HoughLinesP_theta = np.pi/90,
                              HoughLinesP_threshold = 0,
                              HoughLinesP_minLineLength = 10,
                              HoughLinesP_maxLineGap = 2,
                              return_summary = True)

misc.printd(result)