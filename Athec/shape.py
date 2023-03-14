import os, math, random
import cv2
import numpy as np
import scipy.stats
from . import misc

def summary_line(arr):
    arr = np.array(arr)
    arr = arr.flatten()
    n = len(arr)
    if n == 0:
        import warnings
        warnings.warn("No lines are found! Summary statistics are returned as -99999.")
        return([-99999] * 9)
    m = np.mean(arr)
    med = np.median(arr)
    std = np.std(arr)
    min = np.amin(arr)
    max = np.amax(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    skewness = scipy.stats.skew(arr)
    kurto = scipy.stats.kurtosis(arr)
    rlist = [m, med, std, min, max, q1, q3, skewness, kurto]
    return(rlist)

def summary_orientation(arr):
    arr = np.array(arr)
    arr = arr.flatten()
    n = len(arr)
    if n == 0:
        import warnings
        warnings.warn("No lines are found! Summary statistics are returned as -99999.")
        return([-99999] * 2)
    cm = scipy.stats.circmean(arr, high = np.pi/2, low = -np.pi/2)
    cstd = scipy.stats.circstd(arr, high = np.pi/2, low = -np.pi/2)
    rlist = [cm, cstd]
    return(rlist)

def summary_degree(arr):
    arr = np.array(arr)
    arr = arr.flatten()
    n = len(arr)
    if n == 0:
        import warnings
        warnings.warn("No lines are found! Summary statistics are returned as -99999.")
        return([-99999] * 2)
    cm = scipy.stats.circmean(arr, high = 90, low = -90)
    cstd = scipy.stats.circstd(arr, high = 90, low = -90)
    rlist = [cm, cstd]
    return(rlist)

def attr_line_hough_edge(img,
                         save_path = None,
                         horizontal_degree = 10,
                         vertical_degree = 80,
                         HoughLinesP_rho = 1,
                         HoughLinesP_theta = np.pi/90,
                         HoughLinesP_threshold = 0,
                         HoughLinesP_minLineLength = 20,
                         HoughLinesP_maxLineGap = 2,
                         return_summary = False):
    
    edges = misc.read_img_gray(img)
    if isinstance(save_path, str):
        bgr = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR) # so we can draw lines in color

    lines = cv2.HoughLinesP(edges,
                            rho = HoughLinesP_rho,
                            theta = HoughLinesP_theta,
                            threshold = HoughLinesP_threshold,
                            minLineLength = HoughLinesP_minLineLength,
                            maxLineGap = HoughLinesP_maxLineGap) # detect lines

    attributes = ["n_line","n_line_hor","n_line_ver","n_line_slant"]
    if return_summary:
        statnames = ["mean","median","std_dev","min","max","quartile_1","quartile_3","skew","kurtosis"]
        attributes =  attributes[:] + \
            ["_".join(["line_length", var]) for var in statnames] + \
            ["_".join(["line_orientation", var]) for var in statnames] + \
            ["line_orientation_circular_mean","line_orientation_circular_std_dev"] + \
            ["_".join(["line_hor_length", var]) for var in statnames] + \
            ["_".join(["line_ver_length", var]) for var in statnames] + \
            ["_".join(["line_slant_length", var]) for var in statnames]

    if lines is None:
        rlist = [0] * 4
        if return_summary:
            import warnings
            warnings.warn("No lines are detected! Summary statistics are returned as -99999.")
            rlist = [0] * 4 + [-99999] * 48
    else:
        rads = []; degrees = []
        lengths = []; lengths_hor = []; lengths_ver = []; lengths_dyn = []
        for line in lines:
            x1,y1,x2,y2 = line[0]
            length = (  (x1-x2)**2 + (y1-y2)**2  )**0.5
            lengths.append(length)
            rad = math.atan2(y2-y1, x2-x1)
            rads.append(rad)
            degree = math.degrees(rad)
            degrees.append(degree)
            abdegree = abs(degree)
            if abdegree <= horizontal_degree: # horizontal lines
                lengths_hor.append(length)
                if isinstance(save_path, str): bgr = cv2.line(bgr,(x1,y1),(x2,y2),(255,204,0),2) # sky blue
            elif abdegree >= vertical_degree: # vertical lines
                lengths_ver.append(length)
                if isinstance(save_path, str): bgr = cv2.line(bgr,(x1,y1),(x2,y2),(114,128,250),2)# salmon
            else: # slanting lines
                lengths_dyn.append(length)
                if isinstance(save_path, str): bgr = cv2.line(bgr,(x1,y1),(x2,y2),(95,253,239),2) # cornsilk 
                
        rlist = [len(lines),len(lengths_hor),len(lengths_ver),len(lengths_dyn)]
        if return_summary:
            rlist = rlist[:] + summary_line(lengths)[:] + \
                summary_line(degrees)[:] + \
                summary_degree(degrees)[:] + \
                summary_line(lengths_hor)[:] + \
                summary_line(lengths_ver)[:] + \
                summary_line(lengths_dyn)[:]

    d = dict(zip(attributes, rlist))
    if isinstance(save_path, str):
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, bgr)
    return(d)

