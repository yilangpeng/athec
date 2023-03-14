import os, random
import cv2
import numpy as np
from . import misc, box

def tf_saliency_spectral_residual(img, 
                                  save_path = None):
    gray = misc.read_img_gray(img)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(gray)
    saliency_map = (saliency_map * 255).astype("uint8") # return 0-1 float, so convert to 0-255
    if isinstance(save_path, str):
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, saliency_map) 
    return(saliency_map)

def tf_saliency_fine_grained(img, 
                             save_path = None):
    gray = misc.read_img_gray(img)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(gray)
    saliency_map = (saliency_map * 255).astype("uint8") # return 0-1 float, so convert to 0-255
    if isinstance(save_path, str):
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, saliency_map) 
    return(saliency_map)

def attr_complexity_saliency(img, 
                             threshold = 0.7,
                             nblock = 20,
                             return_block = False):
    bw = misc.read_img_gray(img)
    h, w = bw.shape
    tsaliency = np.sum(bw) # total saliency value
    wsaliency = tsaliency/(255*w*h)  # normalized by image size

    windows = misc.split_block(bw, nblock)  # split the saliency map into n*n blocks
    sals = [np.sum(x) for x in windows]  # calculate the saliency for each block
    sals_sorted = np.array(sals)  
    sals_sorted = sals_sorted[np.argsort(-sals_sorted)]  # sort these blocks in the decreasing order
    sals_cum = np.cumsum(sals_sorted)  # get cumulative sum

    nsal = np.argmax(sals_cum >= tsaliency * threshold) + 1 # get the number of blocks that add to the threshold
    ntotal = nblock ** 2
    wblock = nsal / ntotal

    d = {"saliency_total":wsaliency,
         "saliency_block":wblock}

    if return_block:
        attributes = ["saliency_block_" + str(x) for x in range(1, ntotal+1)]
        draw = dict(zip(attributes, sals))
        d.update(draw)
    return(d)

def attr_complexity_saliency_box(img,
                                 save_path = None,
                                 min_perentage = 0.8,
                                 check_interval = 1):
    gray = misc.read_img_gray(img)
    minsize_percent, minbox = box.find_box(gray, m = min_perentage, k = check_interval)
    if isinstance(save_path, str):
        drimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        drimg = cv2.rectangle(drimg,(minbox[1], minbox[0]),(minbox[3], minbox[2]),(0,204,255),4) 
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, drimg)
    d = {"saliency_box_size":minsize_percent,
         "saliency_box_x1":minbox[1],
         "saliency_box_y1":minbox[0],
         "saliency_box_x2":minbox[3],
         "saliency_box_y2":minbox[2]}
    return(d)

def attr_complexity_saliency_consistency(img1, img2,
                                         top_percent = 0.3,
                                         nblock = 5):
    bw1 = misc.read_img_gray(img1)
    bw2 = misc.read_img_gray(img2)

    blocks1 = misc.split_block(bw1, nblock)
    sals1 = np.array( [np.sum(x) for x in blocks1] ) # get saliency in each block
    blocks2 = misc.split_block(bw2, nblock)
    sals2 = np.array( [np.sum(x) for x in blocks2] )

    ntop = int(nblock * nblock * top_percent)  # the number of top blocks  

    sorted_index1 = np.argsort(sals1) # sort the saliency values, return the index of each block in an increasing order
    sorted_index2 = np.argsort(sals2)

    topindex1 = sorted_index1[-ntop:] # get the indices for top blocks (the last ntop elements)
    topindex2 = sorted_index2[-ntop:]

    shareindex = np.intersect1d(topindex1, topindex2) # get blocks that appear in top blocks in both 

    consistency = len(shareindex) / (nblock * nblock)    
    d = {"saliency_consistency":consistency}
    return(d)

def attr_ruleofthirds_centroid(img, 
                               save_path = None):
    bw = misc.read_img_gray(img)
    h, w = bw.shape
    dia = (w**2 + h**2)**0.5
    weight_all = np.sum(bw)

    # get coordinates of the center of mass (CoM), scaled by width/height
    com_x = np.sum([np.sum(bw[:, i]) * (i+1) for i in range(0,w)]) / weight_all
    com_x_s = com_x / w
    
    com_y = np.sum( [np.sum(bw[j, :]) * (j+1) for j in range(0,h)] ) / weight_all
    com_y_s = com_y / h

    # visual balance based on deviation of the CoM
    dcm_x = abs(com_x_s - 1/2) 
    dcm_y = abs(com_y_s - 1/2)

    dcm_c = (dcm_x ** 2 + dcm_y ** 2) ** 0.5


    # rule of thirds based on CoM 
    dist1 = abs(com_x_s - 1/3)
    dist2 = abs(com_x_s - 2/3) # distances to 1/3 and 2/3 vertical lines
    dist_x = min(dist1, dist2)

    dist3 = abs(com_y_s - 1/3)
    dist4 = abs(com_y_s - 2/3) # distances to 1/3 and 2/3 horizontal lines
    dist_y = min(dist3, dist4)

    dist5 = (dist1**2 + dist3**2) ** 0.5
    dist6 = (dist1**2 + dist4**2) ** 0.5  
    dist7 = (dist2**2 + dist3**2) ** 0.5  
    dist8 = (dist2**2 + dist4**2) ** 0.5  # distances to points of intersection
    dist_d = min(dist5, dist6, dist7, dist8) 

    d = {"com_x":com_x_s,
         "com_y":com_y_s,
         "balance_com_ver":dcm_x,
         "balance_com_hor":dcm_y,
         "balance_com_center":dcm_c,
         "rot_com_ver1":dist1,
         "rot_com_ver2":dist2,
         "rot_com_ver":dist_x,
         "rot_com_hor1":dist3,
         "rot_com_hor2":dist4,
         "rot_com_hor":dist_y,
         "rot_com_int1":dist5,
         "rot_com_int2":dist6,
         "rot_com_int3":dist7,
         "rot_com_int4":dist8,
         "rot_com_int":dist_d}

    if isinstance(save_path, str):
        line_color = (192,192,192)
        com_color = (0,204,255)
        thick = int(min(w, h)/100)
        drimg = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        drimg = cv2.line(drimg, (int(1/3*w), 0), (int(1/3*w), h), color=line_color, thickness=thick)
        drimg = cv2.line(drimg, (int(2/3*w), 0), (int(2/3*w), h), color=line_color, thickness=thick)
        drimg = cv2.line(drimg, (0, int(1/3*h)), (w, int(1/3*h)), color=line_color, thickness=thick)
        drimg = cv2.line(drimg, (0, int(2/3*h)), (w, int(2/3*h)), color=line_color, thickness=thick)
        drimg = cv2.circle(drimg, (int(com_x), int(com_y)), radius= int(thick*3), color=com_color, thickness=-1)  # negative thickness means a filled circle

        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, drimg)
    return(d)

def attr_ruleofthirds_band(img, 
                           save_path = None):
    bw = misc.read_img_gray(img)
    h, w = bw.shape
    weight_all = np.sum(bw, axis=(0,1)) # get all weights
    weight_r3_v1 = np.sum(bw[:, int(w*1/4):int(w*5/12)]) / weight_all # weight in the strips
    weight_r3_v2 = np.sum(bw[:, int(w*7/12):int(w*3/4)]) / weight_all
    weight_r3_v = max(weight_r3_v1, weight_r3_v2)

    weight_r3_h1 = np.sum(bw[int(h*1/4):int(h*5/12), :]) / weight_all
    weight_r3_h2 = np.sum(bw[int(h*7/12):int(h*3/4), :]) / weight_all
    weight_r3_h = max(weight_r3_h1, weight_r3_h2)

    weight_r3_is1 = np.sum(bw[int(h*1/4):int(h*5/12), int(w*1/4):int(w*5/12)]) / weight_all # weight in the intersections
    weight_r3_is2 = np.sum(bw[int(h*1/4):int(h*5/12), int(w*7/12):int(w*3/4)]) / weight_all
    weight_r3_is3 = np.sum(bw[int(h*7/12):int(h*3/4), int(w*1/4):int(w*5/12)]) / weight_all
    weight_r3_is4 = np.sum(bw[int(h*7/12):int(h*3/4), int(w*7/12):int(w*3/4)]) / weight_all
    weight_r3_is = max(weight_r3_is1, weight_r3_is2, weight_r3_is3, weight_r3_is4)

    r3_v_bands = [weight_r3_v1, weight_r3_v2, weight_r3_v]
    r3_h_bands = [weight_r3_h1, weight_r3_h2, weight_r3_h]
    r3_intersections = [weight_r3_is1, weight_r3_is2, weight_r3_is3, weight_r3_is4, weight_r3_is]

    d = {"rot_band_ver1":weight_r3_v1,
         "rot_band_ver2":weight_r3_v2,
         "rot_band_ver":weight_r3_v,
         "rot_band_hor1":weight_r3_h1,
         "rot_band_hor2":weight_r3_h2,
         "rot_band_hor":weight_r3_h,
         "rot_band_int1":weight_r3_is1,
         "rot_band_int2":weight_r3_is2,
         "rot_band_int3":weight_r3_is3,
         "rot_band_int4":weight_r3_is4,
         "rot_band_int":weight_r3_is}

    if isinstance(save_path, str):
        band_color = (255, 191, 0) # note OpenCV uses BGR color space
        drimg = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        overlay = drimg.copy()
        overlay = cv2.rectangle(overlay, (0,int(h*1/4)), (w,int(h*5/12)), band_color, -1)
        overlay = cv2.rectangle(overlay, (0,int(h*7/12)), (w,int(h*3/4)), band_color, -1)  
        overlay = cv2.rectangle(overlay, (int(w*1/4),0), (int(w*5/12),h), band_color, -1)  
        overlay = cv2.rectangle(overlay, (int(w*7/12),0), (int(w*3/4),h), band_color, -1)  
        alpha_transparent = 0.5
        drimg = cv2.addWeighted(src1=overlay, alpha=alpha_transparent, src2=drimg, beta=1-alpha_transparent, gamma=0)

        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, drimg)
    return(d)
