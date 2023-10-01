#-*- coding:utf-8 -*-
import os
import time
import torch
import glob
import datetime
import numpy as np
import torch.nn as nn
# dataloader
import datasets_multiple
# nwtwork
from networks.VTCE_Net import VTCE_Net
# RAFT flow
from RAFTcore.raft import RAFT
from networks.resample2d_package.resample2d import Resample2d
# utils
import utils
import cv2
import numpy as np
from utils import Logger, align_to_64, denorm, findLastCheckpoint, findLastCheckpoint_SDSD

# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

class Tester(object):
    def __init__(self, args):
       
        # Model configuration.
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_path = os.path.join(args.save_root_dir, args.version, args.model_save_path)
        self.log_path = os.path.join(args.save_root_dir, args.version, args.log_path)
        self.test_result_path_videoSRC = os.path.join(args.save_root_dir, args.version, args.test_syn)
        self.test_result_path_SDSD = os.path.join(args.save_root_dir, args.version, args.test_SDSD)
        
        # Build the model
        self.build_model() 
        
    def test_SDSD(self):
        """ Test Low Light Enhancement Video ."""
        self.load_pretrained_model(self.args.pretrained_model)
        self.G.eval()

        print("======================================= start testing =========================================")
        test_ids = [line.rstrip('\n') for line in open(self.args.SDSD_list_dir + 'test_list.txt')]

        for test_id in test_ids:
            self.test_result_path_frames = self.test_result_path_SDSD + "/frames/%s"%test_id
            self.test_result_path_video  = self.test_result_path_SDSD + "/video/%s"%test_id
            self.test_result_path_mask = self.test_result_path_SDSD + "/mask/%s"%test_id
            self.test_result_path_warp = self.test_result_path_SDSD + "/warp/%s"%test_id
            self.test_result_path_flo = self.test_result_path_SDSD + "/flo/%s"%test_id
            
            if not os.path.isdir(self.test_result_path_frames):
                os.makedirs(self.test_result_path_frames)
            if not os.path.isdir(self.test_result_path_video):
                os.makedirs(self.test_result_path_video)
            if not os.path.isdir(self.test_result_path_mask):
                os.makedirs(self.test_result_path_mask)
            if not os.path.isdir(self.test_result_path_warp):
                os.makedirs(self.test_result_path_warp)
            if not os.path.isdir(self.test_result_path_flo):
                os.makedirs(self.test_result_path_flo)

            frame_list = sorted(glob.glob(os.path.join(self.args.SDSD_dir, test_id, '*.png')))
            for t in range(2, len(frame_list)-3):
                frame_i0, _ = utils.read_img(os.path.join(self.args.SDSD_dir, test_id, "%05d.png" % (t-2)))
                frame_i1, _ = utils.read_img(os.path.join(self.args.SDSD_dir, test_id, "%05d.png" % (t-1)))
                frame_i2, _ = utils.read_img(os.path.join(self.args.SDSD_dir, test_id, "%05d.png" % (t)))
                frame_i3, _ = utils.read_img(os.path.join(self.args.SDSD_dir, test_id, "%05d.png" % (t+1)))
                frame_i4, _ = utils.read_img(os.path.join(self.args.SDSD_dir, test_id, "%05d.png" % (t+2)))
                frame_i5, _ = utils.read_img(os.path.join(self.args.SDSD_dir, test_id, "%05d.png" % (t+3)))
                
                flow_warping = Resample2d().cuda()

                with torch.no_grad():
                    frame_i0 = utils.img2tensor(frame_i0).cuda()
                    frame_i1 = utils.img2tensor(frame_i1).cuda()
                    frame_i2 = utils.img2tensor(frame_i2).cuda()
                    frame_i3 = utils.img2tensor(frame_i3).cuda()
                    frame_i4 = utils.img2tensor(frame_i4).cuda()
                    frame_i5 = utils.img2tensor(frame_i5).cuda()
                   
                    frame_input  = torch.cat((frame_i0.detach(), frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach(), frame_i5.detach()), 1) 
                    frame_first, frame_second = self.G(frame_input) # frame_out
                
                    # cycle self-regularization optical flow 
                    #_, flow_up_21 = self.Raft_model(frame_second, frame_first, iters=12, test_mode=True)
                    #warp_frame1 = flow_warping(frame_first, flow_up_21)                
                    #noc_mask1 = torch.exp(- 50 * torch.sum(frame_second - warp_frame1, dim=1).pow(2) ).unsqueeze(1)
                    #_, flow_up_12 = self.Raft_model(frame_first, frame_second, iters=12, test_mode=True)
                    _, flow_up_12 = self.Raft_model(frame_first, frame_second, iters=12, test_mode=True)
                    warp_frame2 = flow_warping(frame_second, flow_up_12)                
                    noc_mask2 = torch.exp(- 50 * torch.sum(frame_first - warp_frame2, dim=1).pow(2) ).unsqueeze(1)
                    
                    frame_first_pred = utils.tensor2img(frame_first) 
                    frame_second_pred = utils.tensor2img(frame_second) 
                    warp_frame2 = utils.tensor2img(warp_frame2) 
                    noc_mask2 = utils.tensor2img(noc_mask2) 
                    
                    # # map flow to rgb image [2, 512, 960]
                    flow_up_12 = flow_up_12[0].permute(1,2,0).cpu().numpy()
                    flow_up_12 = flow_to_image(flow_up_12)
                    cv2.imwrite(os.path.join(self.test_result_path_flo, '%05d.png'%t), flow_up_12)
                    
                    utils.save_img(frame_first_pred, os.path.join(self.test_result_path_frames, '%05d.png'%t))
                    utils.save_img(frame_second_pred, os.path.join(self.test_result_path_frames, '%05d.png'%(t+1)))
                    utils.save_img(warp_frame2, os.path.join(self.test_result_path_warp, '%05d.png'%t))
                    utils.save_img(noc_mask2, os.path.join(self.test_result_path_mask, '%05d.png'%t))
                    #utils.save_img(flow_up_12, os.path.join(self.test_result_path_flo, '%05d.png'%t))
                    
       
    def test_videoSRC(self):
        """ Test Low Light Enhancement Video ."""
        self.load_pretrained_model(self.args.pretrained_model)
        self.TD_model.eval()
        self.Raft_model.eval()

        print("======================================= start testing =========================================")
        test_ids = [line.rstrip('\n') for line in open(self.args.videoSRC_list_dir + 'test_list.txt')]

        for test_id in test_ids:
            self.test_result_path_frames = self.test_result_path_videoSRC + "/frames/%s"%test_id
            self.test_result_path_video = self.test_result_path_videoSRC + "/video/%s"%test_id
            self.test_result_path_mask = self.test_result_path_videoSRC + "/mask/%s"%test_id
            self.test_result_path_warp = self.test_result_path_videoSRC + "/warp/%s"%test_id
            
            if not os.path.isdir(self.test_result_path_frames):
                os.makedirs(self.test_result_path_frames)
            if not os.path.isdir(self.test_result_path_video):
                os.makedirs(self.test_result_path_video)
            if not os.path.isdir(self.test_result_path_mask):
                os.makedirs(self.test_result_path_mask)
            if not os.path.isdir(self.test_result_path_warp):
                os.makedirs(self.test_result_path_warp)

            frame_list = sorted(glob.glob(os.path.join(self.args.videoSRC_data_dir, test_id, '*.png')))
            for t in range(2, len(frame_list)-3):
                frame_i0, _ = utils.read_img(os.path.join(self.args.videoSRC_data_dir, test_id, "%05d.png" % (t-2)))
                frame_i1, _ = utils.read_img(os.path.join(self.args.videoSRC_data_dir, test_id, "%05d.png" % (t-1)))
                frame_i2, _ = utils.read_img(os.path.join(self.args.videoSRC_data_dir, test_id, "%05d.png" % (t)))
                frame_i3, _ = utils.read_img(os.path.join(self.args.videoSRC_data_dir, test_id, "%05d.png" % (t+1)))
                frame_i4, _ = utils.read_img(os.path.join(self.args.videoSRC_data_dir, test_id, "%05d.png" % (t+2)))
                frame_i5, _ = utils.read_img(os.path.join(self.args.videoSRC_data_dir, test_id, "%05d.png" % (t+3)))
                
                flow_warping = Resample2d().cuda()

                with torch.no_grad():
                    frame_i0 = utils.img2tensor(frame_i0).cuda()
                    frame_i1 = utils.img2tensor(frame_i1).cuda()
                    frame_i2 = utils.img2tensor(frame_i2).cuda()
                    frame_i3 = utils.img2tensor(frame_i3).cuda()
                    frame_i4 = utils.img2tensor(frame_i4).cuda()
                    frame_i5 = utils.img2tensor(frame_i5).cuda()
                    frame_i0, f_h_pad, f_w_pad = align_to_64(frame_i0, 64)
                    frame_i1, f_h_pad, f_w_pad = align_to_64(frame_i1, 64)
                    frame_i2, f_h_pad, f_w_pad = align_to_64(frame_i2, 64)
                    frame_i3, f_h_pad, f_w_pad = align_to_64(frame_i3, 64)
                    frame_i4, f_h_pad, f_w_pad = align_to_64(frame_i4, 64)
                    frame_i5, f_h_pad, f_w_pad = align_to_64(frame_i5, 64)
                    
                    [b, c, h, w] = frame_i0.shape
                    frame_first_input = torch.cat((frame_i0.detach(), frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach()), 1) # [B, 3*5, 960, 1408]
                    frame_second_input = torch.cat((frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach(), frame_i5.detach()), 1) # [B, 3*5, 960, 1408]
                    frame_first, frame_second = self.TD_model(frame_first_input, frame_second_input)  
                    
                    
                    # cycle self-regularization optical flow 
                    _, flow_up_21 = self.Raft_model(frame_second, frame_first, iters=8, test_mode=True)
                    warp_frame1 = flow_warping(frame_first, flow_up_21)                
                    noc_mask1 = torch.exp(- 50 * torch.sum(frame_second - warp_frame1, dim=1).pow(2) ).unsqueeze(1)

                    _, flow_up_12 = self.Raft_model(frame_first, frame_second, iters=8, test_mode=True)
                    warp_frame2 = flow_warping(frame_second, flow_up_12)                
                    noc_mask2 = torch.exp(- 50 * torch.sum(frame_first - warp_frame2, dim=1).pow(2) ).unsqueeze(1)
                    
                    frame_first = frame_first[:, :, 0:h-f_h_pad, 0:w-f_w_pad] # [1, 3, 918, 1374]
                    frame_second = frame_second[:, :, 0:h-f_h_pad, 0:w-f_w_pad] # [1, 3, 918, 1374]
                    
                    warp_frame1 = warp_frame1[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                    noc_mask1 = noc_mask1[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                    
                    frame_first_pred = utils.tensor2img(frame_first) # [918, 1374, 3]
                    frame_second_pred = utils.tensor2img(frame_second) # [918, 1374, 3]
                    
                    warp_frame1 = utils.tensor2img(warp_frame1) # [918, 1374, 3]
                    noc_mask1 = utils.tensor2img(noc_mask1) # [918, 1374, 3]
                    
                    utils.save_img(frame_first_pred, os.path.join(self.test_result_path_frames, '%05d.png'%t))
                    utils.save_img(frame_second_pred, os.path.join(self.test_result_path_frames, '%05d.png'%(t+1)))
                    
                    utils.save_img(warp_frame1, os.path.join(self.test_result_path_warp, '%05d.png'%t))
                    utils.save_img(noc_mask1, os.path.join(self.test_result_path_mask, '%05d.png'%t))
    
    def build_model(self):
        """Generator: Enhancement models 3D Conv + 2D Conv, Discriminator: """
        self.G = VTCE_Net().to(self.device)
        self.Raft_model = RAFT(self.args).to(self.device)
        if self.args.parallel:
            self.G.to(self.args.gpu_ids[0])    
            self.Raft_model.to(self.args.gpu_ids[0])
            self.G = nn.DataParallel(self.G, self.args.gpu_ids)
            self.Raft_model = nn.DataParallel(self.Raft_model, self.args.gpu_ids)
        print("=== Models have been created ===")
        
        # count network parameters
        if self.args.is_print_network:
            self.count_network_parameters(self.G, 'Enhancement Generator model')
            self.count_network_parameters(self.Raft_model, 'Raft model')
        print("=== Models have been counted ===")
                
        
    def load_pretrained_model(self, resume_epochs):
        checkpoint_path = os.path.join(self.model_save_path, '{}_epoch_{}.pth'.format(self.args.version, resume_epochs))
        if torch.cuda.is_available():
            # save on GPU, load on GPU
            checkpoint = torch.load(checkpoint_path)
            self.G.load_state_dict(checkpoint['G_net'])
            self.G.epoch = resume_epochs
        else:
            # save on GPU, load on CPU
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(checkpoint['G_net'])
            self.G.epoch = resume_epochs
        print("=========== loaded trained models (epochs: {})! ===========".format(resume_epochs))
  
    def count_network_parameters(self, model, name):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([np.prod(p.size()) for p in parameters])
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, N, N / 1e6))    