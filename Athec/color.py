import os, sys, random, warnings
import numpy as np
import cv2
import scipy.stats, scipy.signal, scipy.spatial
import pyemd
from . import misc, colordict

def summary(arr):
    
    arr = arr.flatten()
    m = np.mean(arr)
    med = np.median(arr)
    std = np.std(arr)
    min = np.amin(arr)
    max = np.amax(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    skew = scipy.stats.skew(arr)
    kurto = scipy.stats.kurtosis(arr)
    value, counts = np.unique(arr, return_counts=True)
    ent = scipy.stats.entropy(counts)
    rlist = [m, med, std, min, max, q1, q3, skew, kurto, ent]
    return(rlist)

def summary_two(arr):
    arr = arr.flatten()
    m = np.mean(arr)
    std = np.std(arr)
    rlist = [m, std]
    return(rlist)

def summary_circular_hue(hue):
    hue = hue.flatten()
    hue = hue.astype("float")
    cm = scipy.stats.circmean(hue, high=180, low=0)
    cstd = scipy.stats.circstd(hue, high=180, low=0)
    rlist = [cm, cstd]
    return(rlist)

def attr_RGB(img,
             return_full = False):
    img = misc.read_img_rgb(img) 
    rgbR, rgbB, rgbG = cv2.split(img)
    if return_full:
        rlist = summary(rgbR) + summary(rgbG) + summary(rgbB)
        statnames = ["mean","median","std_dev","min","max","quartile_1","quartile_3","skew","kurtosis","entropy"]
    else:
        rlist = summary_two(rgbR) + summary_two(rgbG) + summary_two(rgbB)
        statnames = ["mean","std_dev"]
    
    channelnames = ["RGB_R", "RGB_G", "RGB_B"]
    attributes = ["_".join([var1, var2]) for var1 in channelnames for var2 in statnames]
    d = dict(zip(attributes, rlist))
    return(d)

def attr_HSV(img,
             return_full = False):
    img = misc.read_img_rgb(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  
    hsvH, hsvS, hsvV = cv2.split(hsv)
    if return_full:
        rlist = summary(hsvH) + summary(hsvS) + summary(hsvV) + summary_circular_hue(hsvH)
        statnames = ["mean","median","std_dev","min","max","quartile_1","quartile_3","skew","kurtosis","entropy"]
    else:
        rlist = summary_two(hsvH) + summary_two(hsvS) + summary_two(hsvV) + summary_circular_hue(hsvH)
        statnames = ["mean","std_dev"]

    channelnames = ["HSV_H", "HSV_S", "HSV_V"]
    attributes = ["_".join([var1, var2]) for var1 in channelnames for var2 in statnames]
    attributes.extend(["HSV_H_circular_mean", "HSV_H_circular_std_dev"])
    d = dict(zip(attributes, rlist))
    return(d)

def attr_HSL(img,
             return_full = False):
    img = misc.read_img_rgb(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)  
    hslH, hslL, hslS = cv2.split(hls)
    if return_full:
        rlist = summary(hslH) + summary(hslS) + summary(hslL) + summary_circular_hue(hslH)
        statnames = ["mean","median","std_dev","min","max","quartile_1","quartile_3","skew","kurtosis","entropy"]
    else:
        rlist = summary_two(hslH) + summary_two(hslS) + summary_two(hslL) + summary_circular_hue(hslH)
        statnames = ["mean","std_dev"]

    channelnames = ["HSL_H", "HSL_S", "HSL_L"]
    attributes = ["_".join([var1, var2]) for var1 in channelnames for var2 in statnames]
    attributes.extend(["HSL_H_circular_mean", "HSL_H_circular_std_dev"])
    d = dict(zip(attributes, rlist))
    return(d)

def attr_XYZ(img,
             return_full = False):
    img = misc.read_img_rgb(img)
    xyz = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)  
    xyzX, xyzY, xyzZ = cv2.split(xyz)
    if return_full:
        rlist = summary(xyzX) + summary(xyzY) + summary(xyzZ)
        statnames = ["mean","median","std_dev","min","max","quartile_1","quartile_3","skew","kurtosis","entropy"]
    else:
        rlist = summary_two(xyzX) + summary_two(xyzY) + summary_two(xyzZ)
        statnames = ["mean","std_dev"]

    channelnames = ["XYZ_X", "XYZ_Y", "XYZ_Z"]
    attributes = ["_".join([var1, var2]) for var1 in channelnames for var2 in statnames]
    d = dict(zip(attributes, rlist))
    return(d)

def attr_Lab(img,
             return_full = False):
    img = misc.read_img_rgb(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)  
    labL, laba, labb = cv2.split(lab)
    if return_full:
        rlist = summary(labL) + summary(laba) + summary(labb)
        statnames = ["mean","median","std_dev","min","max","quartile_1","quartile_3","skew","kurtosis","entropy"]
    else:
        rlist = summary_two(labL) + summary_two(laba) + summary_two(labb)
        statnames = ["mean","std_dev"]

    channelnames = ["Lab_L", "Lab_a", "Lab_b"]
    attributes = ["_".join([var1, var2]) for var1 in channelnames for var2 in statnames]
    d = dict(zip(attributes, rlist))
    return(d)

def tf_grayscale(img,
                 save_path = None):
    rgb = misc.read_img_rgb(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if isinstance(save_path, str):
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, gray)
    return(gray)

def attr_grayscale(img,
                   return_full = False):
    img = misc.read_img_rgb(img) 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if return_full:
        rlist = summary(gray)
        statnames = ["mean","median","std_dev","min","max","quartile_1","quartile_3","skew","kurtosis","entropy"]
    else:
        rlist = summary_two(gray)
        statnames = ["mean","std_dev"]
    
    attributes = ["_".join(["gray", var]) for var in statnames]
    d = dict(zip(attributes, rlist))
    return(d)

def smallest_sublist(alist, target):
    '''Find the smallest sublist in a list with sum greater than a give value'''
    window_sum = 0     # store the current window sum
    left = 0     # store window's starting index
    length = float('inf')    # store the result
    final_left = 0
    final_right = len(alist)     # maintain a sliding window from left to right
    for right in range(len(alist)):     # include current element in the window
        window_sum += alist[right]    # window becomes unstable if its sum becomes more than k
        while window_sum > target and left <= right:
            # update the result if current window's length is less
            # than minimum found so far
            if right - left + 1 < length:
                length = right - left + 1
                final_left = left; final_right = right
            # remove elements from the window's left side till window becomes stable again
            window_sum -= alist[left]
            left = left + 1
    return([length, final_left, final_right])

def attr_contrast_range(img,
                        save_path = None,
                        threshold = 0.90):
    gray = misc.read_img_gray(img)
    hist = np.bincount(gray.ravel(), minlength=256)
    sum = np.sum(hist)
    phist = hist/sum
    minimal = sum * threshold
    min_range, low_idx, high_idx = smallest_sublist(hist, minimal)

    if isinstance(save_path, str):
        from matplotlib import pyplot as plt
        from matplotlib.ticker import PercentFormatter
        color_histogram = 'darkgray'
        color_range = 'lightcoral'

        plt.figure(dpi=200)
        plt.fill_between(range(0,256), 0, phist, color=color_histogram)
        plt.fill_between(range(low_idx, high_idx), 0, phist[low_idx:high_idx], color=color_range)

        plt.xlim([0,256])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().set_ylim(bottom=0)

        for spine in plt.gca().spines.values(): spine.set_visible(False) # remove frames
        plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False)
        
        save_path = misc.make_path(save_path)
        plt.savefig(save_path)
        plt.close()
    
    d = {'contrast_range':min_range,
         'contrast_range_lower':low_idx,
         'contrast_range_upper':high_idx}
    return(d)

def attr_contrast_peak(img,
                       save_path = None,
                       savgol_filter_window_length = 51,
                       savgol_filter_polyorder = 5,
                       savgol_filter_mode = "constant",
                       argrelmax_order = 20):
    gray = misc.read_img_gray(img)
    hist = np.bincount(gray.ravel(), minlength=256)
    sum = np.sum(hist)
    phist = hist/sum

    # apply a smoothing filter to histogram
    smoothed = scipy.signal.savgol_filter(phist, savgol_filter_window_length, savgol_filter_polyorder, mode = savgol_filter_mode)
    # detect peaks
    peaks = scipy.signal.argrelmax(smoothed, order = argrelmax_order)
    peaks = peaks[0]

    if isinstance(save_path, str):
        from matplotlib import pyplot as plt
        from matplotlib.ticker import PercentFormatter
        plt.figure(dpi=200)

        xs = range(0,256)
        plt.fill_between(xs, 0, phist, color='darkgray')
        plt.plot(smoothed,color='midnightblue',linewidth=1)
        plt.scatter(peaks,np.take(smoothed, peaks),marker='^',color='crimson',s=100,zorder=10)

        plt.xlim([0,256])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().set_ylim(bottom=0)

        for spine in plt.gca().spines.values(): spine.set_visible(False) # remove frames
        plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False)

        save_path = misc.make_path(save_path)
        plt.savefig(save_path)
        plt.close()

    npeak = len(peaks)
    if npeak == 0:
        peak_range = -99999
        peak_list = []
        warnings.warn("No peaks are detected! Peak distance is returned as -99999.")
    elif npeak == 1:
        peak_range = -99999
        peak_list = list(peaks)
        warnings.warn("Only one peak is detected! Peak distance is returned as -99999.")
    else:
        peak_range = peaks[-1] - peaks[0]
        peak_list = list(peaks)

    d = {'contrast_n_peak':npeak,
         'contrast_peak_distance':peak_range,
         'contrast_peak_list':peak_list}
    return(d)

def attr_colorful(img):
    rgb = misc.read_img_rgb(img)
    rgb = rgb.astype(float)
    l_rgbR, l_rgbG, l_rgbB = cv2.split(rgb)
    l_rg = l_rgbR - l_rgbG
    l_yb = 0.5*l_rgbR  + 0.5*l_rgbG - l_rgbB
    rg_sd  = np.std(l_rg); rg_mean = np.mean(l_rg)
    yb_sd  = np.std(l_yb); yb_mean = np.mean(l_yb)
    rg_yb_sd = (rg_sd**2 + yb_sd**2)**0.5
    rg_yb_mean = (rg_mean**2 + yb_mean**2)**0.5
    colorful = rg_yb_sd + (rg_yb_mean * 0.3)
    d = {'colorful':colorful}
    return(d)

def attr_colorful_emd(img):
    rgb = misc.read_img_rgb(img)
    h, w = rgb.shape[:2]

    n = 4 # number of bins per channel
    distribution1 = np.empty(n**3)
    distribution1.fill(1/(n**3))    # distribution 1: pixels are evenly distributed in all the bins; the ideal color distribution of a "colorful" image

    rgb4 = rgb//(256/n) # assign each RGB value into bins

    centers = []
    distribution2 = [] 
    for i in range(0, n): 
        for j in range(0, n):
            for k in range(0, n):
                rgb_center = [  (128/n + x*256/n)  for x in [i, j, k]] # center of each bin in RGB color model
                rgb_center = np.array([[rgb_center]]).astype('uint8') # put this pixel as an image and convert to unit8 data type
                luv_center = cv2.cvtColor(rgb_center, cv2.COLOR_RGB2Luv) # center of each bin in LUV color space
                centers.append(  luv_center[0,0,:].astype(float)    ) # attach this center

                npixels = np.sum(np.all(rgb4 == np.array([i,j,k]), axis=2)) # count the number of pixels in this bin in the image
                distribution2.append(npixels)
    distribution2 = np.array(distribution2).astype('float64')
    distribution2 = distribution2/(w*h) # distribution 2: the actual image in the bins

    ntotal = n**3
    distance_matrix = np.zeros([ntotal, ntotal]) # create a distance matrix between all center points
    for i in range(0, ntotal):
        for j in range(0, ntotal):
            distance_matrix[i,j] = scipy.spatial.distance.euclidean(centers[i],centers[j]) 

    emdist = pyemd.emd(distribution1, distribution2, distance_matrix) # calculate the earth mover's distance between two distributions
    colorful_emd = 128 - emdist # small distance between distribution 1 and 2 means the image is more colorful, so reverse the value
    d = {'colorful_emd':colorful_emd}
    return(d)

def attr_color_percentage(img,
                          color_dict = None,
                          save_path = None):
    rgb = misc.read_img_rgb(img)
    h, w = rgb.shape[:2]
    rgb = rgb.astype('int64') # transform to int64 so it can have values larger than 255
    rgb8 = rgb//8 # put values into bins
    r8, g8, b8 = cv2.split(rgb8) # get arrays of R, G, B values in bins
    one = r8 * (32 * 32) + g8 * 32 + b8 # convert to one number
    if color_dict is None:
        color_dict = colordict.color_dict()

    tf = color_dict[one] # get color back; color_dict is a list that is similar to dict: its index is key
    unique, counts = [ list(x) for x in np.unique(tf, return_counts=True) ] # count colors
    color_percents = [0] * 11
    for i in range(0,11):
        if i in unique:
            k = unique.index(i)
            color_percents[i] = counts[k] / (h*w) # get the percentages of each color

    black, blue, brown, gray, green, orange, pink, purple, red, white, yellow = color_percents[:]

    nonbw_colors = np.array([blue, brown, green, orange, pink, purple, red, yellow]) # exclude black, white, and gray
    if np.sum(nonbw_colors) > 0:
        nonbw_colors = nonbw_colors/np.sum(nonbw_colors)
        shannon_e = scipy.stats.entropy(nonbw_colors) # Shannon entropy
        simpson_i = 1 - np.sum([p**2 for p in nonbw_colors]) # Simpson index. Subtracted from 1 so higher value means more diverse hues
    else:
        warnings.warn("All colors are achromatic! Color variety measures are returned as -99999.")
        shannon_e = -99999
        simpson_i = -99999

    d = {'black':black,
         'blue':blue,
         'brown':brown,
         'gray':gray,
         'green':green,
         'orange':orange,
         'pink':pink,
         'purple':purple,
         'red':red,
         'white':white,
         'yellow':yellow,
         'color_shannon':shannon_e,
         'color_simpson':simpson_i}

    if isinstance(save_path, str):
        # create a list of RGB values of eleven colors
        l_colorrgbs = [(0, 0, 0),(0, 0, 255),(128, 102, 64), (128, 128, 128), # black, blue, brown, gray
                       (0, 255, 0),(255, 165, 0),(255,192,203),(128,0,128), # green, orange, pink, purple
                       (255, 0, 0),(255, 255, 255),(255, 255, 0)] # red, white, yellow
        tfimg = np.zeros((h, w, 3), dtype = 'uint8') # create a blank image
        for i in range(0,11):
            mask = (tf == i) # create a mask (True/False)
            tfimg[mask] = l_colorrgbs[i] # assign color based on mask
        bgr = cv2.cvtColor(tfimg, cv2.COLOR_RGB2BGR) # transform to BGR
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, bgr)
    return(d)

def hue_colors(nbin = 20):
    '''get 20 colors for visualization'''
    hsvH = range(int(180/nbin/2), 180, int(180/nbin))
    hsvH = np.array(hsvH)
    hsvS = np.full((nbin,), 255)
    hsvV = np.full((nbin,), 240)
    hsv = np.dstack((hsvH,hsvS,hsvV))
    hsv = hsv.astype("uint8")
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = rgb/255
    rgb = rgb.reshape(nbin, 3)
    rgb = list(map(tuple, rgb))
    return(rgb)

def attr_hue_count(img, 
                   save_path = None,
                   saturation_low = 0.2, 
                   value_low = 0.15, 
                   value_high = 0.95,
                   hue_count_alpha = 0.05):
    rgb = misc.read_img_rgb(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, w, m = hsv.shape[:3]
    hsv_l = hsv.reshape(h*w, m) # reshape to a "list" of HSV values
    hsv_g = hsv_l[(hsv_l[:,1] >= 255*saturation_low) & (hsv_l[:,2]>=255*value_low) & (hsv_l[:,2]<=255*value_high)]
    # select "good" pixels that represent color: saturation > 0.2; 0.15 < value < 0.95
    # H ranges from 0 to 180, S and V from 0 to 255
 
    hgood = hsv_g[:,0] # convert all hue values in a 1-D array
    if len(hgood) > 0:
        bins = np.linspace(0, 180, 20, endpoint=False) # create 20 bins on H value
        digitized = np.digitize(hgood, bins) # put each hue into the bins
        digitized = digitized - 1 # minus 1 since digitized bin index starts from 1
        bin_ns = [len(digitized[digitized == i]) for i in range(0, len(bins))] # count pixels in each hue bin

        n_hgood = len(hgood); n_in_bins = np.sum(bin_ns) # get number of good hue pixels; this two values should be equal
        bin_ps = [x/n_in_bins for x in bin_ns] # calculate percentages

        bin_ns = np.array(bin_ns)
        count_indexes = np.where(bin_ns > hue_count_alpha * max(bin_ns))[0] # get "countable" bins larger than threshold = alpha * the largest bin
        hue_count = len(count_indexes) # get hue count
    else:
        hue_count = 0

    if isinstance(save_path, str) and len(hgood) > 0:
        from matplotlib import pyplot as plt
        from matplotlib.ticker import PercentFormatter
        plt.figure(dpi=200)
        plt.bar(range(0,20), bin_ps, color=hue_colors())
        plt.xlim([-0.5,19.5])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().set_ylim(bottom=0)
        plt.gca().set_xticks(range(0,20))
        plt.gca().set_xticklabels(range(1,21))
        plt.axhline(y= hue_count_alpha * max(bin_ps), color='black', linestyle='--') # plot the threshold
        for spine in plt.gca().spines.values(): spine.set_visible(False) # remove frames
        plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False)

        save_path = misc.make_path(save_path)
        plt.savefig(save_path)
        plt.close()

    d = {'hue_count':hue_count}
    return(d)
