import os
from athec import misc, segment

img_folder = os.path.join("image", "original")
resize_folder = os.path.join("image", "resize")
tf_folder = os.path.join("image", "transform")

imgname = "example1.jpg"
img = os.path.join(resize_folder, imgname)

'''
Perform image segmentation with quickshift method.
Return a 2-D array.
save_path (optional, default None): str. If provided, three visualization will be saved. The first visualization assigns random, non-overlapping colors to segments to preserve the segment labels. This visualization is saved with save_path. The second visualization overlaps pre-defined colors over the original image. For this visualization, "overlap" will be added to the file name. The third visualization replaces each segment with its average color. For this visualization, "average" will be added to the file name. The last two visualizations do not fully store the segmentation results. See https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.label2rgb
ratio, kernel_siz, max_dist, sigma (optional): Parameters for quickshit segmentation. See https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.quickshift
'''
segment_qs = segment.tf_segment_quickshift(img,
                                           save_path = os.path.join(tf_folder, "segment quickshift", imgname),
                                           ratio = 1,
                                           kernel_siz = 5,
                                           max_dist = 10,
                                           sigma = 0)

'''
Calculate visual complexity based on segmentation.
Return:
(1) the number of segments
(2) the number of segments that are larger than several thresholds
(3) the relative sizes of the largest segments.
segment_thresholds (default [0.05, 0.02, 0.01]): a list of floats. The thresholds the segments should be larger than.
top_areas (default 5): int. The number of the largest segments.
'''
result = segment.attr_complexity_segment(segment_qs,
                                         segment_thresholds = [0.05, 0.02, 0.01],
                                         top_areas = 5)
misc.printd(result)

'''
This function can also take the file path of the segmentation image as the input.
'''
segment_path = os.path.join(tf_folder, "segment quickshift", imgname + '.png')
result = segment.attr_complexity_segment(segment_path)
misc.printd(result)

'''
Perform segmentation with normalized cut method.
Return a 2-D array.
save_path (optional, default None): str. If provided, three visualization will be saved. The first visualization assigns random, non-overlapping colors to segments to preserve the segment labels. This visualization is saved with save_path. The second visualization overlaps pre-defined colors over the original image. For this visualization, "overlap" will be added to the file name. The third visualization replaces each segment with its average color. For this visualization, "average" will be added to the file name. The last two visualizations do not fully store the segmentation results. See https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.label2rgb
km_n_segments, km_compactness, rag_sigma, nc_thresh, nc_num_cuts, nc_max_edge: parameters for normalized cut segmentation. See https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_ncut.html
'''
segment_nc = segment.tf_segment_normalized_cut(img,
                                                save_path = os.path.join(tf_folder, "segment normalized cut", imgname),
                                                km_n_segments = 100,
                                                km_compactness = 30,
                                                rag_sigma = 100,
                                                nc_thresh = 0.001, 
                                                nc_num_cuts = 10, 
                                                nc_max_edge = 1.0)

result = segment.attr_complexity_segment(segment_nc)
misc.printd(result)

