# -*- coding: utf-8 -*-
import sys, os, sys, random, shutil
import cv2
import numpy as np
from skimage import io
from PIL import Image
from scipy.spatial import distance

def attr_size(imgpath):
    img = cv2.imread(imgpath)
    filesize = os.path.getsize(imgpath)
    h, w = img.shape[:2]
    ar = w/h; size = w*h
    dia = (w**2 + h**2)**0.5
    rlist = [filesize, w, h, ar, size, dia]
    return(rlist)

def attr_mode(img):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if not os.path.isfile(imgpath):
        print('no file exists')
        return(None)
    ofilesize = os.path.getsize(imgpath)
    try:
        im = Image.open(imgpath)
    except:  
        print("can't open with PIL")
        return(None)
    mode = im.mode
    print("Image mode",mode)
    img = im.convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img.astype("uint8")

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

    imgname = os.path.basename(imgpath)
    imgnamebase = os.path.splitext(imgname)[0]
    imggsavepath = os.path.join(tffolder, imgnamebase + '.jpg')
    misc.create_path(imggsavepath)
    cv2.imwrite(imggsavepath, resized)
    return([mode, ofilesize, ow, oh, w, h])

def tf_resize(img_path, resize_folder, 
              max_w=-99, max_h=-99, max_side=-99, max_size=-99):
    ofilesize = os.path.getsize(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img.astype("uint8")

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

    imgname = os.path.basename(imgpath)
    imgnamebase = os.path.splitext(imgname)[0]
    imggsavepath = os.path.join(tffolder, imgnamebase + '.jpg')
    misc.create_path(imggsavepath)
    cv2.imwrite(imggsavepath, resized)
    return([mode, ofilesize, ow, oh, w, h])

    

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

def create_save_path2(img, save_folder, save_subfolder = None, save_name = None):
    if isinstance(save_name, str):
        imgname = save_name if save_name.lower().endswith(".png") else save_name + '.png'
    elif isinstance(img, str) and img:
        imgbasename = os.path.basename(img)
        imgname = imgbasename if imgbasename.lower().endswith(".png") else imgbasename + '.png'
    else:
        import string, random
        letters = string.ascii_letters
        imgname = ''.join(random.choice(letters) for i in range(10))+ '.png'

    if isinstance(save_subfolder, str) and save_subfolder:
        save_path = os.path.join(save_folder, save_subfolder, imgname)
    else:
        save_path = os.path.join(save_folder, imgname)
    filepath_folder = os.path.dirname(save_path)
    if not os.path.exists(filepath_folder): os.makedirs(filepath_folder)
    return(save_path)

def get_img_name(img, save_name = None):
    if isinstance(save_name, str):
        imgname = save_name
    elif isinstance(img, str) and img:
        imgbasename = os.path.basename(img)
        imgname = imgbasename
    else:
        import string, random
        letters = string.ascii_letters
        imgname = ''.join(random.choice(letters) for i in range(10))+ '.png'
    return(imgname)

def create_save_path(img, save_folder, save_subfolder, save_name = None):
    if isinstance(save_name, str):
        imgname = save_name if save_name.lower().endswith(".png") else save_name + '.png'
    elif isinstance(img, str) and img:
        imgbasename = os.path.basename(img)
        imgname = imgbasename if imgbasename.lower().endswith(".png") else imgbasename + '.png'
    else:
        import string, random
        letters = string.ascii_letters
        imgname = ''.join(random.choice(letters) for i in range(10))+ '.png'

    if isinstance(save_subfolder, str) and save_subfolder:
        save_path = os.path.join(save_folder, save_subfolder, imgname)
    else:
        save_path = os.path.join(save_folder, imgname)
    filepath_folder = os.path.dirname(save_path)
    if not os.path.exists(filepath_folder): os.makedirs(filepath_folder)
    return(save_path)

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

def create_path(filepath):
    if not os.path.exists(filepathfolder): os.makedirs(filepathfolder)
    filepath = os.path.join()

def copy_file(filepath1, filepath2):
    filepathfolder2 = os.path.dirname(filepath2) 
    if not os.path.exists(filepathfolder2): os.makedirs(filepathfolder2)
    shutil.copy2(filepath1, filepath2)

def save_list_to_txt(wlist, filesavepath):
    create_path(filesavepath)
    slist = [str(x) for x in wlist]
    jlist = '\t'.join(slist)
    with open(filesavepath, "a") as resultf:
        resultf.write(jlist + '\n')


def zip_print(l1, l2):
    for x in list(zip(l1, l2)): 
        print(x[0], x[1]) 

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

def rotate(image, angle): # rotate counter clockwise
    # this function is copyed from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/    
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def juxtapose_img(imgpath, save_folder = None, tf_methods = None):
    imgbasename = os.path.basename(imgpath)
    imgname = imgbasename if imgbasename.lower().endswith(".png") else imgbasename + '.png'

    rgb = read_img_rgb(imgpath)

    jux = rgb
    h, w = rgb.shape[:2]
    inspect(jux)
    if save_folder and tf_methods:
        for tf_method in tf_methods:
            tf_path = os.path.join(save_folder, tf_method, imgname)
            print(tf_path)
            tf_img = read_img_rgb(tf_path)
            inspect(tf_img)
            tf_img = resize_img(tf_img, to_h = h)
            inspect(tf_img)
            jux = np.concatenate((jux, tf_img), axis=1)
    
    tf_method = " ".join(tf_methods)
    save_path = create_save_path(imgpath, save_folder, tf_method)
    jux = cv2.cvtColor(jux, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, jux)

def index_inner(nblock):
    ntotal = nblock ** 2
    index_all = np.array(range(0, ntotal))
    quotient = index_all // nblock
    remainder = index_all % nblock
    keep1 = ( quotient != 0 ) # remove first row
    keep2 = ( quotient != (nblock - 1 ) ) # remove last row
    keep3 = ( remainder != 0 ) # remove first column
    keep4 = ( remainder != (nblock - 1) ) # remove last column
    inner = ( keep1 * keep2 * keep3 * keep4 > 0 )
    index_keep = np.where(inner)
    return(index_keep)
