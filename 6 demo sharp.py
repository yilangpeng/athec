import os
from athec import misc, sharp

img_folder = os.path.join("image", "original")
resize_folder = os.path.join("image", "resize")
tf_folder = os.path.join("image", "transform")

imgname = "example1.jpg"
img = os.path.join(resize_folder, imgname)

'''
Calculate sharpness as the standard deviation of Laplacian.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
'''
result = sharp.attr_sharp_laplacian(img,
                                    save_path = os.path.join(tf_folder, "sharp laplacian", imgname))
misc.printd(result)

'''
Calculate sharpness with fast Fourier transform.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
'''
result = sharp.attr_sharp_fft(img,
                              save_path = os.path.join(tf_folder, "sharp fft", imgname))
misc.printd(result)

'''
Calculate sharpness as the standard deviation of maximum local variations of all pixels.
save_path (optional, default None): str. If provided, a visualization will be saved to this location.
'''
result = sharp.attr_sharp_mlv(img,
                              save_path = os.path.join(tf_folder, "sharp mlv", imgname))
misc.printd(result)

'''
Calculate measures of depth of field based on sharpness measures after the image is partitioned into 4 × 4 blocks.
Return:
(1) the sharpness blur measure of the four inner blocks, divided by the average sharpness measure of all the blocks.
(1) summary statistics of sharpness measures of all the blocks.
(2) sharpness measure in each block, if return_raw is set to True.
sharp_method (optional, default "laplacian"): str ("laplacian" or "fft" or "mlv") or function name. The function to calculate the blur measure for each block.
return_summary (default False): bool. If set to True, summary statistics of sharpness measures will be returned.
return_block (default False): bool. If set to True, the sharpness measures of all the blocks will be returned.
'''
result = sharp.attr_dof_block(img,
                              sharp_method = sharp.attr_sharp_fft, 
                              return_summary = True,
                              return_block = True)
misc.printd(result)
