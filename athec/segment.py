#encoding=utf-8
import os, random
import numpy as np
import skimage.io, skimage.segmentation, skimage.measure, skimage.future
from . import misc

def save_label(rgb, label, save_path):
    # visualization 1: assign random colors to segments
    nlabel = np.amax(label)
    random_colors = random.sample(range(1, 16777215), nlabel)
    random_colors = np.array(random_colors)
    label = label - 1  # as skimage.measure.label start from 1
    colored = random_colors[label]

    l_rgbB = colored % 256
    l_rgbG = ( colored/256 ) % 256
    l_rgbR = ( colored / (256^2) ) % 256
    rand = np.dstack((l_rgbR,l_rgbG,l_rgbB))

    save_path1 = misc.make_path(save_path)
    skimage.io.imsave(save_path1, rand)
    
    # visualization 2: overlay colors onto original image
    overlay = skimage.color.label2rgb(label, rgb, kind='overlay')
    save_path2 = misc.make_path(save_path, 'overlay')
    skimage.io.imsave(save_path2, overlay)

    # visualization 3: average colors in each segment
    avg = skimage.color.label2rgb(label, rgb, kind='avg')
    save_path3 = misc.make_path(save_path, 'average')
    skimage.io.imsave(save_path3, avg)

def tf_segment_quickshift(img,
                          save_path = None,
                          ratio = 1,
                          kernel_siz = 5,
                          max_dist = 10,
                          sigma = 0):
    rgb = misc.read_img_rgb(img)
    seg = skimage.segmentation.quickshift(rgb, ratio=ratio, kernel_size=kernel_siz,
                                  max_dist=max_dist, sigma=sigma)
    # see https://scikit-image.org/docs/dev/api/skimage.segmentation.html
    label, n_seg = skimage.measure.label(seg, background=99999999, return_num=True) # by default 0-valued pixels are considered as background, so set background value to 999999

    if isinstance(save_path, str):
        save_label(rgb, label, save_path)
    return(label)

def tf_segment_normalized_cut(img,
                              save_path = None,
                              km_n_segments = 100,
                              km_compactness = 30,
                              rag_sigma = 100,
                              nc_thresh = 0.001, 
                              nc_num_cuts = 10, 
                              nc_max_edge = 1.0):
    rgb = misc.read_img_rgb(img)

    seg_km = skimage.segmentation.slic(rgb,
                               n_segments = km_n_segments,
                               compactness = km_compactness,
                               start_label = 1)

    rag = skimage.future.graph.rag_mean_color(rgb,
                               seg_km, mode='similarity', 
                               sigma=rag_sigma)  

    seg = skimage.future.graph.cut_normalized(seg_km, rag, 
                               thresh=nc_thresh, num_cuts=nc_num_cuts, 
                               max_edge=nc_max_edge)

    label, n_seg = skimage.measure.label(seg, background=99999999, return_num=True) # by default 0-valued pixels are considered as background, so set background value to 999999

    if isinstance(save_path, str):
        save_label(rgb, label, save_path)
    return(label)

def attr_complexity_segment(img,
                            segment_thresholds = [0.05, 0.02, 0.01],
                            top_areas = 5):
    if isinstance(img, str):
        rgb = misc.read_img_rgb(img)
        img = rgb.astype('int64')
        l_rgbR, l_rgbG, l_rgbB = img[:, :, 0], img[:, :, 1], img[:, :, 2] # get R, G, B values
        seg = 65536 * l_rgbR + 256 * l_rgbG + l_rgbB; # convert each RGB to a unique integer (hex); max: 16777215
    elif isinstance(img, np.ndarray) and len(img.shape) == 2:
        seg = img
    else:
        raise ValueError("Need to provide correct input image")

    h, w = seg.shape
    seg, n_seg = skimage.measure.label(seg, background=99999999, return_num=True) # by default 0-valued pixels are considered as background, so set background value to 999999
    unique_counts = np.unique(seg,return_counts=True)
    counts = unique_counts[1] # first array is label value, second array is count
    counts = np.flip(np.sort(counts)) # sort in a decreasing order
    nseg = len(counts) # number of segments

    l_nseg = []
    for segment_threshold in segment_thresholds:
        nseg_threshold = np.count_nonzero(counts >= (w*h*segment_threshold) ) # number of segments larger than the threshold
        l_nseg.append(nseg_threshold)

    if len(counts) >= top_areas:
        counts = counts[:top_areas]
    sizes = counts / (w*h)
    sizes = sizes.tolist() # convert from array to list

    if len(sizes) < top_areas:
        import warnings
        warnings.warn("The number of segments is fewer than the number of top segments requested. The remaining sizes are returned as 0.")
        sizes.extend( [0] * (top_areas - len(sizes)) ) # if requested number of areas to return > the number of segments, add 0

    rlist = [nseg] + l_nseg[:] + sizes[:]
    attributes = ["n_segments"] + \
            ["n_segments_" + str(x) for x in segment_thresholds] + \
            ["segment_top_" + str(x) for x in range(1, top_areas + 1)]
    d = dict(zip(attributes, rlist))
    return(d)