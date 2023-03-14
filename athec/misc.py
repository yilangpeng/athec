# -*- coding: utf-8 -*-
import os, sys, random, shutil, pprint
import cv2
import numpy as np
from PIL import Image, ImageFile

def create_path(filepath):
    filepathfolder = os.path.dirname(filepath) 
    if not os.path.exists(filepathfolder): os.makedirs(filepathfolder)

def make_path(filepath, suffix = None):
    imgbasename = os.path.basename(filepath)
    if isinstance(suffix, str):
        imgbasename = imgbasename + "." + suffix
    imgname = imgbasename if imgbasename.lower().endswith(".png") else imgbasename + '.png'
    filepathfolder = os.path.dirname(filepath)
    if not os.path.exists(filepathfolder): 
        os.makedirs(filepathfolder)
    newfilepath = os.path.join(filepathfolder, imgname)
    return(newfilepath)

def copy_file(filepath1, filepath2):
    filepathfolder2 = os.path.dirname(filepath2) 
    if not os.path.exists(filepathfolder2): os.makedirs(filepathfolder2)
    shutil.copy2(filepath1, filepath2)

def save_list_to_txt(wlist, filesavepath):
    '''Save a list to .txt.'''
    create_path(filesavepath)
    slist = [str(x) for x in wlist]
    jlist = '\t'.join(slist)
    with open(filesavepath, "a") as resultf:
        resultf.write(jlist + '\n')

def save_dict_to_txt(d, filesavepath, save_keys = False):
    '''Save a dictionary to .txt.'''
    create_path(filesavepath)
    if save_keys: save_list_to_txt(list(d.keys()), filesavepath) 
    save_list_to_txt(list(d.values()), filesavepath) 

def printd(d):
    pprint.pprint(d, sort_dicts=False)

def attr_size(imgpath):
    img = cv2.imread(imgpath)
    filesize = os.path.getsize(imgpath)
    h, w = img.shape[:2]
    ar = w/h; size = w*h
    dia = (w**2 + h**2)**0.5
    wfilesize = filesize / size
    d = {"file_size":filesize,
         "width":w,
         "height":h,
         "aspect_ratio":ar,
         "image_size":size,
         "diagonal":dia,
         "file_size_s":wfilesize}
    return(d)

def attr_mode(imgpath):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        im = Image.open(imgpath)
        mode = im.mode
    except:  
        import warnings
        warnings.warn("can't open with PIL. Returned as -99999.")
        mode = -99999
    d = {"image_mode":mode}
    return(d)

def tf_resize(img_path, resize_path, 
              max_w=-99, max_h=-99, max_side=-99, max_size=-99):
    ofilesize = os.path.getsize(img_path)
    img = read_img_bgr(img_path)

    h, w = img.shape[:2]
    oh, ow = img.shape[:2]
    ar = w/h
    if max_side > 0:
        if w > h:
            if w > max_side:
                w = max_side
                h = w/ar
        else:
            if h > max_side:
                h = max_side
                w = h * ar
    if max_w > 0:
        if w > max_w:
            w = max_w
            h = w/ar
    if max_h > 0:
        if h > max_h:
            h = max_h
            w = h * ar
    if max_size > 0:
        if w*h > max_size:
            h = (max_size/ar) ** 0.5
            w = (max_size*ar) ** 0.5
    w = int(w+0.5); h = int(h+0.5)
    resized = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)

    create_path(resize_path)
    cv2.imwrite(resize_path, resized)
    filesize = os.path.getsize(resize_path)

    d = {"file_size_original":ofilesize,
         "width_original":ow,
         "height_original":oh,
         "file_size_resize":filesize,
         "width_resize":w,
         "height_resize":h}
    return(d)

def tf_binary(img, 
              save_path = None,
              threshold = None):
    gray = read_img_gray(img)
    if isinstance(threshold, int):
        ret, binarized = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        ret, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if isinstance(save_path, str):
        save_path = make_path(save_path)
        cv2.imwrite(save_path, binarized)
    return(binarized)

def read_img_rgb(img):
    if isinstance(img, str):
        try:
            img = cv2.imread(img)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return(img)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return(img)
            else:
                raise ValueError("Need to provide correct input image")
        except:
            raise ValueError("Can't read file")
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            return(img)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)    
            return(img)
        else:
            raise ValueError("Need to provide correct input image")
    else:
        raise ValueError("Need to provide image path or image array")
    return(None)

def read_img_bgr(img):
    if isinstance(img, str):
        try:
            img = cv2.imread(img)
            if len(img.shape) == 3:
                return(img)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return(img)
            else:
                raise ValueError("Need to provide correct input image")
        except:
            raise ValueError("Can't read file")
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            return(img)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return(img)
        else:
            raise ValueError("Need to provide correct input image")
    else:
        raise ValueError("Need to provide image path or image array")
    return(None)

def read_img_gray(img):
    if isinstance(img, str):
        try:
            img = cv2.imread(img)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return(img)
            elif len(img.shape) == 2:
                return(img)
            else:
                raise ValueError("Need to provide correct input image")
        except:
            raise ValueError("Can't read file")
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return(img)
        elif len(img.shape) == 2:
            return(img)
        else:
            raise ValueError("Need to provide correct input image")
    else:
        raise ValueError("Need to provide image path or image array")

def inspect(array, length = 10, window = [8, 8]):
    print("="*30,"-----start of inspection-----")
    array = np.array(array)
    print("shape",array.shape)
    print("data type",array.dtype)
    print("mean median min max", np.mean(array), np.median(array), np.amin(array), np.amax(array))
    if len(array.shape) == 1:
        array = array[:length]
    elif len(array.shape) == 2:
        array = array[:window[0],:window[1]]
    elif len(array.shape) == 3:
        array = array[:window[0],:window[1],:]
    else:
        print('not supported!'); return(None)
    print("window\n",array,"\n","-----end of inspection-----","="*30)

def split_block(bw, nblock):
    sp1d = np.array_split(bw, nblock, axis=0) #split the array into a list of nblock subarrays along 0 axis
    sp2d = [np.array_split(x, nblock, axis=1) for x in sp1d]  #  flat = [x for sublist in sp2d for x in sublist] # flatten the nested list
    windows = [x for sublist in sp2d for x in sublist] # flatten the nested list and get sum in each block#    print(sals)
    return(windows)

