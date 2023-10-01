# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
import cv2
import glob
import datetime


def calc_ssim(Output_img, GT_img):   
    test_Y = False  # True: test Y channel only; False: test RGB channels
    
    im_Gen = Output_img 
    im_GT = GT_img 

    if test_Y and im_GT.shape[2] == 3: # evaluate on Y channel in YCbCr color space
        im_GT_in = bgr2ycbcr(im_GT)
        im_Gen_in = bgr2ycbcr(im_Gen)
    else:
        im_Gen_in = im_Gen
        im_GT_in = im_GT

    ssim_result = ssim_skimage(im_GT_in , im_Gen_in , multichannel=True, data_range=255)
    return ssim_result

def read_img_raw(filename, grayscale=0):
    ## read image and convert to RGB in [0, 1]
    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Image %s does not exist" %filename)
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise Exception("Image %s does not exist" %filename)
        img = img[:, :, ::-1] ## BGR to RGB
    img = np.float32(img) / 65535.0
    return img

def read_img(filename, grayscale=0):
    ## read image and convert to RGB in [0, 1]
    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Image %s does not exist" %filename)
        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise Exception("Image %s does not exist" %filename)
        img = img[:, :, ::-1] ## BGR to RGB
    img = np.float32(img) / 255.0
    return img


if __name__ == '__main__':
    im_Gen = read_img("/home/lingyu/code/Enhancement_Video/ECCV2022_expand/src1_semi_framework/data/SDSD_CUHK/videoSDSD_low_resize/outdoor_pair1/00004.png") 
    im_GT = read_img("/home/lingyu/code/Enhancement_Video/ECCV2022_expand/src1_semi_framework/data/SDSD_CUHK/videoSDSD_normal_resize/outdoor_pair1/00004.png") 
    
    ssim_result = calc_ssim(im_Gen * 255. ,  im_GT * 255.)
    
    print("========output ssim", ssim_result) # 0.47049