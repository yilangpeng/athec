import os, random, math
import numpy as np
import cv2
import scipy.spatial, scipy.stats
import misc, box

def tf_edge_canny(img,
                  save_path = None,
                  otsu_ratio = 0.5,
                  thresholds = None, 
                  gaussian_blur_kernel = None):
    gray = misc.read_img_gray(img)
    if gaussian_blur_kernel:
        gray = cv2.GaussianBlur(gray, gaussian_blur_kernel, 0)
    if thresholds:
        edge = cv2.Canny(gray, threshold1 = thresholds[0], threshold2 = thresholds[1])
    elif otsu_ratio:
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge = cv2.Canny(gray, threshold1 = (ret * otsu_ratio), threshold2=ret) 
    else:
        raise ValueError("Need to provide either thresholds or otsu_ratio")

    if isinstance(save_path, str):
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, edge)
    return(edge)

def attr_complexity_edge(img, 
                         n_random = 1000):
    edge = misc.read_img_gray(img)
    edge_threshold = 0
    epoints = np.argwhere(edge > edge_threshold) # return indices of edge points on each dimension of the image array

    h, w = edge.shape
    dia = (h**2 + w**2)**0.5 # length of diagonal
    etotal = len(epoints)
    edensity = etotal/(h*w) # percentage of edge points
    if etotal > 0:
        random.seed(42)
        if isinstance(n_random, int):
            epoints = random.sample(list(epoints), min(n_random, etotal)) #get a random sample of edge points instead of using all
        dists = scipy.spatial.distance.pdist(epoints, 'euclidean')
        edist = np.mean(dists)/dia # compute average distance among edge points
    else:
        edist = -99999
    d = {"edge_density":edensity,
         "edge_distance":edist}
    return(d)

def attr_complexity_edge_box(img,
                             save_path = None,
                             min_perentage = 0.9,
                             check_interval = 1):
    gray = misc.read_img_gray(img)
    minsize_percent, minbox = box.find_box(gray, m = min_perentage, k = check_interval)
    if isinstance(save_path, str):
        drimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        drimg = cv2.rectangle(drimg,(minbox[1], minbox[0]),(minbox[3], minbox[2]),(0,204,255),4) # OpenCV uses x, y coordinates similar to PIL (but "coordinates" in arrays are reversed)
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, drimg)
    d = {"edge_box_size":minsize_percent,
         "edge_box_x1":minbox[1],
         "edge_box_y1":minbox[0],
         "edge_box_x2":minbox[3],
         "edge_box_y2":minbox[2]}
    return(d)
