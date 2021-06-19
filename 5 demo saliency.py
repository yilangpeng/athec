import os, sys
from athec import misc, saliency

img_folder = os.path.join("image", "original")
resize_folder = os.path.join("image", "resize")
tf_folder = os.path.join("image", "transform")

imgname = "example1.jpg"
img = os.path.join(resize_folder, imgname)
 
'''
Perform saliency detection with spectral residual method. See https://docs.opencv.org/3.4/d8/d65/group__saliency.html and https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/
Return a 2-D array.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
'''
saliency_spectral = saliency.tf_saliency_spectral_residual(img,
                                                           save_path = os.path.join(tf_folder, "saliency spectral", imgname))

'''
Perform saliency detection with fine grained method. See https://docs.opencv.org/3.4/d8/d65/group__saliency.html and https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/ 
Return a 2-D array.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
'''
saliency_fine = saliency.tf_saliency_fine_grained(img, 
                                                  save_path = os.path.join(tf_folder, "saliency fine", imgname))

'''
Binarize an grayscale image. See https://www.pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
Return a 2-D array.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
threshold (default None): int. The threshold for binarization. Pixels above the threshold will be set to 255 and pixels below it will be set to 0. If this argument is not provided, the function will use the Ostu method to automatically find the threshold.
'''
saliency_spectral_bin = misc.tf_binary(saliency_spectral,
                                           save_path = os.path.join(tf_folder, "saliency spectral binary", imgname),
                                           threshold = 60)

saliency_fine_bin = misc.tf_binary(saliency_fine,
                                       save_path = os.path.join(tf_folder, "saliency fine binary", imgname),
                                       threshold = 60)

'''
Calculate visual complexity based on saliency.
Return:
(1) the total saliency values after saliency detection (normalized by image size).
(2) the number of image blocks that add to a certain percentage (i.e., threshold) of total saliency values after the image is partitioned into n × n (i.e., nblock) blocks.
(3) the saliency values in each image block.
threshold (default 0.7): float. See above.
nblock (default 20): int. See above. 
return_block (default False): bool. If set to True, the function will return the saliency values in each block.
'''
result = saliency.attr_complexity_saliency(saliency_spectral,
                                           threshold = 0.7,
                                           nblock = 10,
                                           return_block = True)

misc.printd(result)

'''
Calculate visual complexity based on the relative size of a minimal bounding box that contains a certain percentage of saliency values.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
see demo edge.py
'''
result = saliency.attr_complexity_saliency_box(saliency_spectral, 
                                 save_path = os.path.join(tf_folder, "bounding box saliency spectral", imgname),
                                 min_perentage = 0.9,
                                 check_interval = 1)
misc.printd(result)

'''
Calculate visual complexity based on the consistency between two saliency maps. This method first divides each saliency map into n x n (i.e., nblock) blocks. This method defines the top image blocks (i.e., top_percent) with the highest saliency values as "salient blocks." The percentage of overlapping salient blocks between two saliency maps measures the consistency between them.
nblock (default 5): int. See above.
top_percent (default 0.6): float. See above.
'''
result = saliency.attr_complexity_saliency_consistency(saliency_spectral, saliency_fine,
                                                       top_percent = 0.6,
                                                       nblock = 5)

misc.printd(result)

'''
Find the center of mass (CoM) of a grayscale image and calculate measures of visual balance and rule of thirds.
Return:
(1) the coordinates of CoM, weighted by image width and height, respectively.
(2) its distances to the central vertical line, the central horizontal line, and the center of the image.
(3) its distances to the 1/3 and 2/3 vertical lines and the minimal distance; its distances to the 1/3 and 2/3 horizontal lines and the minimal distance; and its distances to the four intersections of the thirds lines and the minimal distance.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
'''

result = saliency.attr_ruleofthirds_centroid(saliency_spectral, 
                               save_path = os.path.join(tf_folder, "ruleofthirds centroid saliency spectral", imgname) )
misc.printd(result)

'''
Calculate measures of rule of thirds based on saliency values that fall within thirds bands and intersections.
Return:
(1) saliency weights in the two vertical thirds bands and the maximal of the two.
(2) saliency weights in the two horizontal thirds bands and the maximal of the two.
(3) saliency weights in the four intersection rectangles and the maximal of the four.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
'''

result = saliency.attr_ruleofthirds_band(saliency_spectral, 
                                         save_path = os.path.join(tf_folder, "ruleofthirds band saliency spectral", imgname) )

misc.printd(result)
