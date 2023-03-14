import os
import cv2
import numpy as np
from . import misc

def size_of_box(box):
    y1, x1, y2, x2 = box
    return (x2-x1+1)*(y2-y1+1)

def value_of_box(box, table):
    y1, x1, y2, x2 = box
    return table[y2,x2] + table[y1,x1] - table[y2,x1] - table[y1,x2]

def find_box(bw, # get a box that bounds a certain percentage of all edge points
             m = 0.90, 
             k = 1): # interval for checking pixels
    h, w = bw.shape
    table = np.zeros((h,w), dtype=np.int)  # create an empty table

    table[0,0] = bw[0,0] # create summed area table (note: in numpy, x and y are exchanged)
    for x in range(1,w):
        table[0,x] = bw[0,x] + table[0,x-1]
    for y in range(1,h):  
        table[y,0] = bw[y,0] + table[y-1,0]
    for y in range(1,h):
        for x in range(1,w):
            table[y,x] = bw[y,x] - table[y-1,x-1] + table[y,x-1] + table[y-1,x]

    total = table[h-1, w-1] 
    if total > 0:
        minvalue = int(total * m) # contain at least this value

        minbox = [0, 0, h-1, w-1]  # create the minimal bounding box, starting from the whole image
        # sequence: y1, x1, y2, x2
        minsize = h*w  # stores value for the size of the minimal bounding box

        for y1 in range(0, h, k):    # find the minimal bounding box that contains minvalue
            min_y2 = h
            while value_of_box((y1, 0, min_y2-1, w - 1), table) >= minvalue:
                min_y2 = min_y2 - 1
            if min_y2 == h: break
            for y2 in range(min_y2, h, k):
                x2 = w - 1
                while value_of_box( (y1, 0, y2, x2 - 1), table ) >= minvalue:
                    x2 = x2 - 1
                for x1 in range(0, w, k):
                    while value_of_box( (y1, x1, y2, x2), table) < minvalue:
                        x2 = x2 + 1
                        if x2 == w: break;
                    if x2 == w: break
                    current_box = (y1, x1, y2, x2)
                    if size_of_box(current_box) <= minsize:
                        minbox = [y1, x1, y2, x2]
                        minsize = size_of_box(minbox)

        minsize_percent = minsize/(h*w)
    else:
        minsize_percent = -99999
        minbox = -99999
    return([minsize_percent, minbox])

def attr_complexity_box(img, 
                        save_path = None,
                        min_perentage = 0.9,
                        check_interval = 1):
    gray = misc.read_img_gray(img)
    minsize_percent, minbox = find_box(gray, m = min_perentage, k = check_interval)

    if isinstance(save_path, str):
        drimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        drimg = cv2.rectangle(drimg,(minbox[1], minbox[0]),(minbox[3], minbox[2]),(0,204,255),4)
        # OpenCV uses x, y coordinates similar to PIL (but "coordinates" in arrays are reversed)
        save_path = misc.make_path(save_path)
        cv2.imwrite(save_path, drimg) # save image

    d = {"box_size":minsize_percent,
         "box_x1":minbox[1],
         "box_y1":minbox[0],
         "box_x2":minbox[3],
         "box_y2":minbox[2]}
    return(d)
