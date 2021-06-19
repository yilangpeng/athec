import os, math
import cv2
import numpy as np
import misc
import scipy.stats

def attr_sharp_laplacian(img,
                         save_path = None):
    gray = misc.read_img_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur = np.std(lap)

    if isinstance(save_path, str):
        lap = cv2.convertScaleAbs(lap)
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, lap)
    d = {"sharp_laplacian": blur}
    return(d)

def attr_sharp_fft(img,
                   save_path = None,
                   magnitude_threshold = 5):
    gray = misc.read_img_gray(img)
    h, w = gray.shape[:2]
    gray255 = gray/255
    f = np.fft.fft2(gray255) # apply Fast Fourier Transform
    fshift = np.fft.fftshift(f) # put in the center
    mags = np.abs(fshift) # get magnitudes
    nmag = np.count_nonzero(mags > magnitude_threshold) # only count frequencies > theta
    blur = nmag/(w*h) # normalize by image size

    if isinstance(save_path, str):
        mag_spectrum = 20 * np.log(mags)
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, mag_spectrum)
    d = {"sharp_fft": blur}
    return(d)

def attr_sharp_mlv(img,
                   save_path = None):
    bw = misc.read_img_gray(img)
    h, w = bw.shape
    m = bw.astype(float) # default is uint8 with range 0-255
    m1 = np.zeros(shape=(h, w)) # create eight images that are 1-pixel away from the original image
    m2 = np.zeros(shape=(h, w))
    m3 = np.zeros(shape=(h, w)) 
    m4 = np.zeros(shape=(h, w))
    m5 = np.zeros(shape=(h, w)) 
    m6 = np.zeros(shape=(h, w))
    m7 = np.zeros(shape=(h, w)) 
    m8 = np.zeros(shape=(h, w))
    m1[0:h-1, 0:w-1] = m[1:h  , 1:w  ]
    m2[0:h-1, 0:w  ] = m[1:h  , 0:w  ]
    m3[0:h-1, 1:w  ] = m[1:h  , 0:w-1]
    m4[0:h  , 1:w  ] = m[0:h  , 0:w-1]
    m5[1:h  , 1:w  ] = m[0:h-1, 0:w-1]
    m6[1:h  , 0:w  ] = m[0:h-1, 0:w  ]
    m7[1:h  , 0:w-1] = m[0:h-1, 1:w  ]
    m8[0:h  , 0:w-1] = m[0:h  , 1:w  ]
    d1 = np.abs(m-m1) # get absolute differences
    d2 = np.abs(m-m2)
    d3 = np.abs(m-m3)
    d4 = np.abs(m-m4)
    d5 = np.abs(m-m5)
    d6 = np.abs(m-m6)
    d7 = np.abs(m-m7)
    d8 = np.abs(m-m8)
    mlvmap = np.maximum.reduce([d1,d2,d3,d4,d5,d6,d7,d8]) # get max difference
    mlvmap = mlvmap[1:h-1,1:w-1] # remove borders
    mlvmap = mlvmap.astype('int64')
    if isinstance(save_path, str):
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, mlvmap)
    blur = np.std(mlvmap) # get standard deviation
    d = {"sharp_mlv": blur}
    return(d)

def summary_sharp(arr):
    arr = np.array(arr)
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

def attr_dof_block(img,
             sharp_method = "laplacian",
             return_summary = False,
             return_block = False):
    nblock = 4; ntotal = nblock ** 2
    gray = misc.read_img_gray(img)
    windows = misc.split_block(gray, nblock)

    if isinstance(sharp_method, str):
        if sharp_method == "laplacian": sharp_method = attr_sharp_laplacian
        if sharp_method == "fft": sharp_method = attr_sharp_fft
        if sharp_method == "mlv": sharp_method = attr_sharp_mlv

    sharps = [  [*sharp_method(x).values()][0] for x in windows] # get blur measures for each block
    sharps_ar = np.array(sharps)

    index_keep = np.array([5,6,9,10]) # get inner blocks
    inners = sharps_ar[index_keep]

    dof_inner = np.mean(inners) / np.mean(sharps) # depth of field measure
    rlist = [dof_inner]; attributes = ["dof_inner"]
    if return_summary:
        summary = summary_sharp(sharps)
        rlist = rlist[:] + summary[:]
        statnames = ["mean","median","std_dev","min","max",
                     "quartile_1","quartile_3","skew","kurtosis"]
        attributes.extend( [ "sharp_block_" + var for var in statnames] )

    if return_block:
        rlist = rlist[:] + sharps[:]
        attributes.extend( ["sharp_block_" + str(x) for x in range(1, ntotal+1)]  ) 
    d = dict(zip(attributes, rlist))
    return(d)
