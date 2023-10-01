# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import cv2
import glob
import datetime


def calc_psnr(Output_img, GT_img):
    test_Y = False  

    im_Gen = Output_img 
    im_GT = GT_img 
    
    if test_Y and im_GT.shape[2] == 3:  
        im_Gen_in = bgr2ycbcr(im_Gen)
        im_GT_in = bgr2ycbcr(im_GT)
    else:
        im_Gen_in = im_Gen
        im_GT_in = im_GT
    
    psnr_result = calculate_psnr(im_GT_in , im_Gen_in )
    return psnr_result

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
 

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

def my_psnr(img1, img2):
   mse = np.mean( (img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    im_Gen = read_img("/home/lingyu/code/Enhancement_Video/ECCV2022_expand/src1_semi_framework/data/SDSD_CUHK/videoSDSD_low_resize/outdoor_pair1/00004.png") 
    im_GT = read_img("/home/lingyu/code/Enhancement_Video/ECCV2022_expand/src1_semi_framework/data/SDSD_CUHK/videoSDSD_normal_resize/outdoor_pair1/00004.png") 
    
    psnr_result = my_psnr(im_Gen ,  im_GT )
    
    print("========output PSNR", psnr_result) # 17.0059
    