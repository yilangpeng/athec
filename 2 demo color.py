import os
from athec import misc, color, colordict

img_folder = os.path.join("image", "original")
resize_folder = os.path.join("image", "resize")
tf_folder = os.path.join("image", "transform")

imgname = "example1.jpg"
img = os.path.join(resize_folder, imgname)

"""
Calculate summary statistics of RGB channels.
return_full (optional, default False): bool. If set to True, ten summary statistics will be returned. Otherwise only mean and standard deviation are returned.
"""
result = color.attr_RGB(img, return_full = True)
misc.printd(result)

"""
Calculate summary statistics of HSV channels.
"""
result = color.attr_HSV(img)
misc.printd(result)

"""
Calculate summary statistics of HSL channels.
"""
result = color.attr_HSL(img)
misc.printd(result)

"""
Calculate summary statistics of XYZ channels
"""
result = color.attr_XYZ(img)
misc.printd(result)

"""
Calculate summary statistics of Lab channels
"""
result = color.attr_Lab(img)
misc.printd(result)

"""
Calculate summary statistics of grayscale channel
"""
result = color.attr_grayscale(img)
misc.printd(result)

"""
Convert the image into a grayscale image.
Return a 2-D array.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
"""
color.tf_grayscale(img,
                   save_path = os.path.join(tf_folder, "grayscale", imgname))

"""
Calculate contrast based on a range that covers a certain percentage of the brightness histogram.
Return the range and its lower and upper limits.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
threshold (optional, default 0.9): float. The percentage the range covers.
"""
result = color.attr_contrast_range(img,
                                   save_path = os.path.join(tf_folder, "contrast range", imgname),
                                   threshold = 0.90)
misc.printd(result)

"""
Calculate contrast based on peak detection on the brightness histogram.
Return the number of peaks, the largest gap between peaks, and all the detected peaks.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
savgol_filter_window_length, savgol_filter_polyorder, savgol_filter_mode (optional): for peak detection, the function first applies a Savitzky-Golay filter to the brigtness histogram. These three parameters correspond to window_length, polyorder, and mode in scipy.signal.savgol_filter. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
argrelmax_order (optional): the function uses scipy.signal.argrelmax to find peaks on the filtered histogram. This parameter corresponds to order in scipy.signal.argrelmax. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html
"""
result = color.attr_contrast_peak(img,
                                  save_path = os.path.join(tf_folder, "contrast peak", imgname),
                                  savgol_filter_window_length = 51,
                                  savgol_filter_polyorder = 5,
                                  savgol_filter_mode = "constant",
                                  argrelmax_order = 20)

misc.printd(result)

"""
Calculate colorfulness based on the formula in Hasler and Suesstrunk (2003)
"""
result = color.attr_colorful(img)
misc.printd(result)

"""
Calculate colorfulness based on the distance between two color distributions (Datta et al., 2006)
"""
result = color.attr_colorful_emd(img)
misc.printd(result)

"""
Get the color dictionary for attr_color_percentage.
"""
cd = colordict.color_dict()

"""
Calculate percentages of eleven specific colors and color variety measures based on color percentages (excluding black, white, and gray).
color_dict (optional, default None): if not provided, the function will automatically import the dictionary using colordict.color_dict().
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
"""
result = color.attr_color_percentage(img, 
                                     color_dict = cd,
                                     save_path = os.path.join(tf_folder, "color percentage", imgname))
misc.printd(result)

"""
Calculate color variety based on hue count formula in Ke et al. (2006).
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
saturation_low (optional, default 0.2): float. The lower limit for saturation. Pixels with saturation below the limit are discarded.
value_low (optional, default 0.15): float. The lower limit for value. Pixels with value below the limit are discarded.
value_high (optional, default 0.95): float. The upper limit for value. Pixels with value above the limit are discarded.
hue_count_alpha (optional, default 0.05): float. Alpha in the hue count formula.
"""

result = color.attr_hue_count(img, 
                              save_path = os.path.join(tf_folder, "hue count", imgname),
                              saturation_low = 0.2, 
                              value_low = 0.15, 
                              value_high = 0.95,
                              hue_count_alpha = 0.05)
misc.printd(result)
