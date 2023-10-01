#-*- coding:utf-8 -*-
import os
import time
import torch
import glob
import datetime
import itertools
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import *

from datasets_multiple import *
from losses import *
import pytorch_ssim
# nwtwork
from networks.VTCE_Net import VTCE_Net
# RAFT flow
from RAFTcore.raft import RAFT
from networks.resample2d_package.resample2d import Resample2d
# utils
import utils
from utils import Logger, denorm, align_to_64, findLastCheckpoint, findLastCheckpoint_SDSD
from metrics.CalcPSNR import calc_psnr
from metrics.CalcSSIM import calc_ssim

class Trainer(object):
    def __init__(self, loaders, args):
        # data loader
        self.loaders = loaders
        # Model configuration.
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_path = os.path.join(args.save_root_dir, args.version, args.model_save_path)
        self.log_path = os.path.join(args.save_root_dir, args.version, args.log_path)          
      
        # Build the model and tensorboard.
        self.build_model()
        if self.args.use_tensorboard:
            self.build_tensorboard()

    def train(self):
        """ Train Semi-framework ."""
        self.fetcher = InputFetcher(self.loaders.ref)
        self.train_steps_per_epoch = len(self.loaders.ref) # 83
        
        self.model_save_step = int(self.args.model_save_epoch * self.train_steps_per_epoch)
        
        # define loss functions 
        criterionGAN = GANLoss(self.args.adv_loss_type, tensor=torch.cuda.FloatTensor) # GAN loss
        vgg_model = VGG19_Extractor(output_layer_list=[3,8,13,22]).to(self.device)     # perceptual loss
        criterion_ssim = pytorch_ssim.SSIM(window_size = 11)                           # ssim loss
        criterion_l1 = nn.L1Loss(size_average=True)                                    # L1 loss
        criterion_char = CharbonnierLoss()                                             # CharbonnierLoss
        criterion_edge = EdgeLoss()                                                    # edge loss
        
        # start from scratch or trained models
        if self.args.pretrained_model:
            start_step = int(self.args.pretrained_model * self.train_steps_per_epoch)
            self.load_pretrained_model(self.args.pretrained_model)
        else:
            start_step = 0
        
        # start training
        print("======================================= start training =======================================")
        self.start_time = time.time()
        total_steps = int(self.args.total_epochs * self.train_steps_per_epoch) 
        pbar = tqdm(total=total_steps, desc='Train epoches', initial=start_step)
        for step in range(start_step, total_steps):
            
            # define empty list
            self.frame_low_syn = []        # syn low frames
            self.filename_low_syn = []     # syn low filename
            self.frame_normal_syn = []     # syn normal frames
            self.filename_normal_syn = []  # syn normal filename
            
            self.frame_low_SDSD = []       # real low frames
            self.filename_low_SDSD = []    # real low filename
            self.frame_normal_SDSD = []    # real normal frames
            self.filename_normal_SDSD = [] # real normal filename   
            
            # model train
            self.G.train()
            flow_warping = Resample2d().cuda()
            
            # data iter
            input = next(self.fetcher)            
            self.x_syn, self.y_syn, self.x_SDSD, self.y_SDSD = input.x_syn, input.y_syn, input.x_SDSD, input.y_SDSD
            self.name_x_syn, self.name_y_syn, self.name_x_SDSD, self.name_y_SDSD = input.name_x_syn, input.name_y_syn, input.name_x_SDSD, input.name_y_SDSD
            
            # assert syn image file path and real image file path 
            assert self.name_x_syn[0][0].split("/")[-2] == self.name_y_syn[0][0].split("/")[-2]                                       # assert syn video name
            assert self.name_x_syn[0][0].split("/")[-1].split(".png")[0] == self.name_y_syn[0][0].split("/")[-1].split(".png")[0]     # assert syn frame name
            assert self.name_x_SDSD[0][0].split("/")[-2] == self.name_y_SDSD[0][0].split("/")[-2]                                     # assert real video name  
            assert self.name_x_SDSD[0][0].split("/")[-1].split(".png")[0] == self.name_x_SDSD[0][0].split("/")[-1].split(".png")[0]   # assert real frame name
                        
            for t in range(self.args.sample_frames):
                self.frame_low_syn.append(self.x_syn[t].cuda())      # syn low
                self.frame_normal_syn.append(self.y_syn[t].cuda())   # syn normal
                self.frame_low_SDSD.append(self.x_SDSD[t].cuda())    # real low
                self.frame_normal_SDSD.append(self.y_SDSD[t].cuda()) # real normal
            
            # Clear optimizer 
            self.g_optimizer.zero_grad()
            
            # source    
            self.low_syn_1 = self.frame_low_syn[0]
            self.low_syn_2 = self.frame_low_syn[1]
            self.low_syn_3 = self.frame_low_syn[2]
            self.low_syn_4 = self.frame_low_syn[3]
            self.low_syn_5 = self.frame_low_syn[4]
            self.low_syn_6 = self.frame_low_syn[5]
            # source GT
            self.normal_syn_1 = self.frame_normal_syn[0]
            self.normal_syn_2 = self.frame_normal_syn[1]
            self.normal_syn_3_first = self.frame_normal_syn[2]
            self.normal_syn_4_second = self.frame_normal_syn[3]
            self.normal_syn_5 = self.frame_normal_syn[4]
            self.normal_syn_6 = self.frame_normal_syn[5]
            # target
            self.low_real_1 = self.frame_low_SDSD[0]
            self.low_real_2 = self.frame_low_SDSD[1]
            self.low_real_3 = self.frame_low_SDSD[2]
            self.low_real_4 = self.frame_low_SDSD[3]
            self.low_real_5 = self.frame_low_SDSD[4]
            self.low_real_6 = self.frame_low_SDSD[5]
            # target GT
            self.normal_SDSD_1 = self.frame_normal_SDSD[0]
            self.normal_SDSD_2 = self.frame_normal_SDSD[1]
            self.normal_SDSD_3_first = self.frame_normal_SDSD[2]
            self.normal_SDSD_4_second = self.frame_normal_SDSD[3]
            self.normal_SDSD_5 = self.frame_normal_SDSD[4]
            self.normal_SDSD_6 = self.frame_normal_SDSD[5]
            # Concat and Forward
            self.low_syn_input = torch.cat((self.low_syn_1.detach(), self.low_syn_2.detach(), self.low_syn_3.detach(), self.low_syn_4.detach(), self.low_syn_5.detach(), self.low_syn_6.detach()), 1)        # synthetic low-light 6 images
            self.low_real_input = torch.cat((self.low_real_1.detach(), self.low_real_2.detach(), self.low_real_3.detach(), self.low_real_4.detach(), self.low_real_5.detach(), self.low_real_6.detach()), 1)  # realistic low-light 6 images  
            self.normal_syn_pred_3_first, self.normal_syn_pred_4_second = self.G(self.low_syn_input)
            self.normal_real_pred_3_first, self.normal_real_pred_4_second = self.G(self.low_real_input)  
      
            # update D 
            # self.d_optimizer.zero_grad()
            # real_exp_preds = self.D(self.normal_syn_3.detach())      # real normal image  [B, 1, 128, 128], [B, 1, 64, 64], [B, 1, 32, 32], [5, 1, 16, 16], [5, 1, 8, 8]
            # fake_exp_preds = self.D(self.normal_real_pred.detach())  # fake normal image
            # d_gan_loss = self.criterionGAN(real_exp_preds, fake_exp_preds, None, None, for_discriminator=True)
            # d_loss = d_gan_loss 
            # d_loss.backward() 
            # self.d_loss = d_loss.item()
            # self.d_optimizer.step()
            
            # update G
            self.g_optimizer.zero_grad()
            
            # spatial loss
            ## perceptual loss
            self.g_percep_loss_target_3 = F_loss(self.normal_real_pred_3_first, self.normal_SDSD_3_first, vgg_model)
            self.g_percep_loss_target_4 = F_loss(self.normal_real_pred_4_second, self.normal_SDSD_4_second, vgg_model)
            ## CharbonnierLoss
            self.g_char_loss_target_3 = criterion_char(self.normal_real_pred_3_first, self.normal_SDSD_3_first) 
            self.g_char_loss_target_4 = criterion_char(self.normal_real_pred_4_second, self.normal_SDSD_4_second) 
            ## edge loss
            self.g_edge_loss_target_3 = criterion_edge(self.normal_real_pred_3_first, self.normal_SDSD_3_first) 
            self.g_edge_loss_target_4 = criterion_edge(self.normal_real_pred_4_second, self.normal_SDSD_4_second) 
            
            
            self.g_percep_loss =  self.g_percep_loss_target_3 + self.g_percep_loss_target_4
            self.g_char_loss = self.g_char_loss_target_3 + self.g_char_loss_target_4
            self.g_edge_loss = self.g_edge_loss_target_3 + self.g_edge_loss_target_4
            
            self.g_spatial_loss = self.g_percep_loss + self.g_char_loss + 0.05*self.g_edge_loss
         
            # temporal loss, cycle self-regularization optical flow 
            _, flow_up_21_source = self.Raft_model(self.normal_syn_pred_4_second, self.normal_syn_pred_3_first, iters=12, test_mode=True)
            warp_frame1_source = flow_warping(self.normal_syn_pred_3_first, flow_up_21_source)                
            noc_mask1_source = torch.exp(- 50 * torch.sum(self.normal_syn_pred_4_second - warp_frame1_source, dim=1).pow(2) ).unsqueeze(1)
            _, flow_up_12_source = self.Raft_model(self.normal_syn_pred_3_first, self.normal_syn_pred_4_second, iters=12, test_mode=True)
            warp_frame2_source = flow_warping(self.normal_syn_pred_4_second, flow_up_12_source)                
            noc_mask2_source = torch.exp(- 50 * torch.sum(self.normal_syn_pred_3_first - warp_frame2_source, dim=1).pow(2) ).unsqueeze(1)      
            
            
            self.g_temporal_loss_source = criterion_l1(self.normal_syn_pred_3_first * noc_mask2_source, warp_frame2_source * noc_mask2_source) + criterion_l1(self.normal_syn_pred_4_second * noc_mask1_source, warp_frame1_source * noc_mask1_source)
            
            self.g_temporal_loss = self.g_temporal_loss_source
           
            self.overall_loss = self.g_spatial_loss + 1.0 * self.g_temporal_loss
            self.overall_loss.backward()
                    
            ## update      
            self.g_optimizer.step()
      
            ### print loss info and save models
            self.print_info_step(step, total_steps, pbar)

            ### logging using tensorboard
            self.logging_step(step) # no problem
             
            ### learning rate update
            self.learning_rate_decay_step(step, pbar)
            
            ### validation 
            self.model_validation_SDSD_step(step)
            
            pbar.update(1)
            pbar.set_description(f"Train epoch %.2f" % ((step+1.0)/self.train_steps_per_epoch))
        pbar.write("=========== Complete training ===========")
        pbar.close()

    # one frame for fast validation 
    def model_validation_SDSD_step(self, step):
        if (step + 1) % self.train_steps_per_epoch == 0:
            current_epoch = (step + 1) / self.train_steps_per_epoch
      
            PSNR_result, Total_PSNR = 0.0, 0.0
            SSIM_result, Total_SSIM = 0.0, 0.0
            
            self.G.eval()
            print("======================================= start validation =======================================")
            video_list = [line.rstrip('\n') for line in open(self.args.SDSD_list_dir + 'test_list.txt')]
            for v in range(len(video_list)):
                video = video_list[v]
                
                ref_frame_5, ref_name = utils.read_img(os.path.join(self.args.SDSD_ref_dir, video, '00004.png'))
                ref_frame_5 = utils.img2tensor(ref_frame_5).cuda() 
                
                frame_i0, frame_i0_filename = utils.read_img(os.path.join(self.args.SDSD_dir, video, "00002.png" ))
                frame_i1, frame_i1_filename = utils.read_img(os.path.join(self.args.SDSD_dir, video, "00003.png" ))
                frame_i2, frame_i2_filename = utils.read_img(os.path.join(self.args.SDSD_dir, video, "00004.png" ))
                frame_i3, frame_i3_filename = utils.read_img(os.path.join(self.args.SDSD_dir, video, "00005.png" ))
                frame_i4, frame_i4_filename = utils.read_img(os.path.join(self.args.SDSD_dir, video, "00006.png" ))
                frame_i5, frame_i5_filename = utils.read_img(os.path.join(self.args.SDSD_dir, video, "00007.png" ))
       
                with torch.no_grad():
                    frame_i0 = utils.img2tensor(frame_i0).cuda() 
                    frame_i1 = utils.img2tensor(frame_i1).cuda()
                    frame_i2 = utils.img2tensor(frame_i2).cuda()
                    frame_i3 = utils.img2tensor(frame_i3).cuda()
                    frame_i4 = utils.img2tensor(frame_i4).cuda()
                    frame_i5 = utils.img2tensor(frame_i5).cuda()
                    
                    frame_input  = torch.cat((frame_i0.detach(), frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach(), frame_i5.detach()), 1) 
                    val_frame_pred_first, val_frame_pred_second = self.G(frame_input) # frame_out
                    
                    if self.args.use_tensorboard:
                        self.val_images = {}
                        self.val_images['Val/target_pred_reference'] = torch.cat([denorm(val_frame_pred_first.detach().cpu()), denorm(ref_frame_5.detach().cpu())], 3)
                        for tag, image in self.val_images.items():
                            self.logger.images_summary(tag, image, current_epoch)
                    
                    val_frame_pred_first = utils.tensor2img(val_frame_pred_first) 
                    ref_frame_5 = utils.tensor2img(ref_frame_5)
                # Validation SSIM PSNR
                if self.args.is_test_psnr_ssim:
                    SSIM_result = calc_ssim(val_frame_pred_first * 255., ref_frame_5 * 255.)
                    Total_SSIM += SSIM_result
                    PSNR_result = calc_psnr(val_frame_pred_first * 255., ref_frame_5 * 255.)
                    Total_PSNR += PSNR_result
      
            if self.args.is_test_psnr_ssim:
                ave_PSNR =  Total_PSNR / len(video_list)
                ave_SSIM =  Total_SSIM / len(video_list)
                print("Average PSNR :",ave_PSNR, "Average SSIM :",ave_SSIM)
    
    def learning_rate_decay_step(self, step, pbar):
        if (step + 1) % self.train_steps_per_epoch == 0:
            current_epoch = (step + 1) / self.train_steps_per_epoch
            self.lr_scheduler_g.step(epoch=current_epoch)
            for param_group in self.g_optimizer.param_groups:
                pbar.write("====== Epoch: {:>3d}/{}, Learning rate(lr) of Encoder(E) and Generator(G): [{}], ".format(((step + 1) // self.train_steps_per_epoch), self.args.total_epochs, param_group['lr']), end='')
                
    def logging_step(self, step):
        self.loss = {}
        self.images = {}
        ## loss visulization
        self.loss['Source_Target/Total'] = self.overall_loss.item()
        ## source loss
        self.loss['Source/temporal_loss'] = self.g_temporal_loss_source.item() 
        ## target loss
        self.loss['Target/percep_loss'] = self.g_percep_loss.item() 
        self.loss['Target/char_loss'] = self.g_char_loss.item() 
        self.loss['Target/edge_loss'] = self.g_edge_loss.item()        
        
        ## image visulization
        self.images['Train_source/syn_low_pred_reference'] = torch.cat([denorm(self.low_syn_3.cpu()), denorm(self.normal_syn_pred_3_first.detach().cpu()), denorm(self.normal_syn_3_first.cpu())], 3)
        self.images['Train_target/real_low_pred_reference'] = torch.cat([denorm(self.low_real_3.cpu()), denorm(self.normal_real_pred_3_first.detach().cpu()), denorm(self.normal_SDSD_3_first.cpu())], 3)
        
        if (step+1) % self.args.log_step == 0:            
            if self.args.use_tensorboard:
                for tag, value in self.loss.items():
                    self.logger.scalar_summary(tag, value, step+1)
                for tag, image in self.images.items():
                    self.logger.images_summary(tag, image, step+1)
                
    def print_info_step(self, step, total_steps, pbar):
        current_epoch = (step + 1) / self.train_steps_per_epoch
        if (step + 1) % self.args.info_step == 0:
            elapsed_num = time.time() - self.start_time
            elapsed = str(datetime.timedelta(seconds=elapsed_num))
            pbar.write("Elapse:{:>.12s}, D_Step:{:>4d}/{}, G_Step:{:>4d}/{}, G_loss:{:>.4f}, G_percep_loss:{:>.4f}, G_l1_loss:{:>.4f}, G_temporal_loss:{:>.4f}".format(elapsed, step + 1, total_steps, (step + 1), total_steps, self.overall_loss, self.g_percep_loss, self.g_char_loss, self.g_temporal_loss)) 
            #pbar.write("Elapse:{:>.12s}, D_Step:{:>4d}/{}, G_Step:{:>4d}/{}, G_loss:{:>.4f}, G_percep_loss:{:>.4f}, G_l1_loss:{:>.4f}".format(elapsed, step + 1, total_steps, (step + 1), total_steps, self.overall_loss, self.g_percep_loss, self.g_l1_loss)) 
                    
        # save models
        if (step + 1) % self.model_save_step == 0:
            if self.args.parallel:
                if torch.cuda.device_count() > 1:
                    checkpoint = {
                    "G_net": self.G.module.state_dict(),
                    "epoch": current_epoch,
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict()
                    }
            else:
                checkpoint = {
                    "G_net": self.G.state_dict(),
                    "epoch": current_epoch,
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "lr_scheduler_g": self.lr_scheduler_g.state_dict()
                }
            torch.save(checkpoint, os.path.join(self.model_save_path, '{}_epoch_{}.pth'.format(self.args.version, int(current_epoch))))
            pbar.write("======= Save model checkpoints into {} ======".format(self.model_save_path))    
    
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

        # init network
        if self.args.init_type:
           self.init_weights(self.G, init_type=self.args.init_type, gain=0.02)
        # load pretrained Raft model
        if self.args.use_pretrained_raft:
            self.Raft_model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(self.args.raft_model_path).items()})
            print("=== Pretrained Raft Model have been Loaded ===")

        # optimizer
        if self.args.optimizer_type == 'adam':
            # Adam optimizer            
            self.g_optimizer = torch.optim.Adam(params=self.G.parameters(), lr=self.args.lr_init, betas=[self.args.beta1, self.args.beta2],  weight_decay=self.args.weight_decay) # [ {'params': self.G.parameters()} ]
        elif self.args.optimizer_type == 'rmsprop':
            # RMSprop optimizer
            self.g_optimizer = torch.optim.RMSprop(params=self.G.parameters(), lr=self.args.g_lr, alpha=self.args.alpha)
        else:
            raise NotImplementedError("=== Optimizer [{}] is not found ===".format(self.args.optimizer_type))   
        
        # learning rate decay
        if self.args.lr_decay:
            def lambda_rule(epoch):
                return 1.0 - max(0, epoch + 1 - self.args.lr_num_epochs_decay) / self.args.lr_decay_ratio
            self.lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule)
            print("=== Set learning rate decay policy for Generator(G) and Discriminator(D) ===")
              
    def load_pretrained_model(self, resume_epochs):
        checkpoint_path = os.path.join(self.model_save_path, '{}_epoch_{}.pth'.format(self.args.version, resume_epochs))
        if torch.cuda.is_available():
            # save on GPU, load on GPU
            checkpoint = torch.load(checkpoint_path)
            self.G.load_state_dict(checkpoint['G_net'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.G.epoch = resume_epochs
        else:
            # save on GPU, load on CPU
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(checkpoint['G_net'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            self.lr_scheduler_g.load_state_dict(checkpoint['lr_scheduler_g'])
            self.G.epoch = resume_epochs
        print("=========== loaded trained models (epochs: {})! ===========".format(resume_epochs))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)
    
    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,   0.0)
        print("=== Initialize network with [{}] ===".format(init_type))
        net.apply(init_func)

    def count_network_parameters(self, model, name):
        """Count network parameters only for requires_grad=True."""
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([np.prod(p.size()) for p in parameters])
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, N, N / 1e6))    
        
    
    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # one frame for fast validation 
    def model_validation_syn(self, epoch):
        current_epoch = epoch
        PSNR_result, Total_PSNR = 0.0, 0.0
        SSIM_result, Total_SSIM = 0.0, 0.0
        
        self.Enhance_model.eval()
        print("======================================= start validation =======================================")
        video_list = [line.rstrip('\n') for line in open(self.args.videoSRC_list_dir + 'test_list.txt')]

        for v in range(len(video_list)):
            video = video_list[v]
            
            ref_frame_5, ref_name = utils.read_img(os.path.join(self.args.videoSRC_ref_data_dir, video, '00005.png'))
            ref_frame_5 = utils.img2tensor(ref_frame_5).cuda() 
            
            frame_i0, frame_i0_filename = utils.read_img(os.path.join(self.args.videoSRC_data_dir, video, "00002.png" ))
            frame_i1, frame_i1_filename = utils.read_img(os.path.join(self.args.videoSRC_data_dir, video, "00003.png" ))
            frame_i2, frame_i2_filename = utils.read_img(os.path.join(self.args.videoSRC_data_dir, video, "00004.png" ))
            frame_i3, frame_i3_filename = utils.read_img(os.path.join(self.args.videoSRC_data_dir, video, "00005.png" ))
            frame_i4, frame_i4_filename = utils.read_img(os.path.join(self.args.videoSRC_data_dir, video, "00006.png" ))
            frame_i5, frame_i5_filename = utils.read_img(os.path.join(self.args.videoSRC_data_dir, video, "00007.png" ))
   
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
       
                ref_frame_5, f_h_pad, f_w_pad = align_to_64(ref_frame_5, 64) 
                [b, c, h, w] = frame_i0.shape
                
                frame_first_input  = torch.cat((frame_i0.detach(), frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach()), 1) 
                frame_second_input = torch.cat((frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach(), frame_i5.detach()), 1) 
                val_first_frame_pred, val_second_frame_pred = self.Enhance_model(frame_first_input, frame_second_input) # [:,:,100:356, 400:656]  
                                
                # crop padding
                val_second_frame_pred = val_second_frame_pred[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                ref_frame_5 = ref_frame_5[:, :, 0:h-f_h_pad, 0:w-f_w_pad]      
                         

                if self.args.use_tensorboard:
                    self.val_images = {}
                    self.val_images['Val/val_frame_pred'] = denorm(val_second_frame_pred.detach().cpu())
                    self.val_images['Val/Ref_frame'] = denorm(ref_frame_5.detach().cpu())
                    for tag, image in self.val_images.items():
                        self.logger.images_summary(tag, image, epoch)
                val_second_frame_pred = utils.tensor2img(val_second_frame_pred) 
                ref_frame_5 = utils.tensor2img(ref_frame_5)

            # Validation SSIM PSNR
            if self.args.is_test_psnr_ssim:
                SSIM_result = calc_ssim(val_second_frame_pred * 255.0, ref_frame_5 * 255.0)
                Total_SSIM += SSIM_result
                PSNR_result = calc_psnr(val_second_frame_pred * 255.0, ref_frame_5 * 255.0)
                Total_PSNR += PSNR_result

        if self.args.use_tensorboard:
            self.val_Metric = {} 
            ave_PSNR =  Total_PSNR / len(video_list)
            ave_SSIM =  Total_SSIM / len(video_list)
            self.val_Metric['Val/PSNR'] =  ave_PSNR
            self.val_Metric['Val/SSIM'] =  ave_SSIM
            print("Average PSNR :",ave_PSNR, "Average SSIM :",ave_SSIM)
            for tag, value in self.val_Metric.items():
                self.logger.scalar_summary(tag, value, epoch)
                   
    # one frame for fast validation 
    def model_validation_SDSD(self, epoch):
        current_epoch = epoch
        PSNR_result, Total_PSNR = 0.0, 0.0
        SSIM_result, Total_SSIM = 0.0, 0.0
        
        self.Enhance_model.eval()
        print("======================================= start validation =======================================")
        video_list = [line.rstrip('\n') for line in open(self.args.SDSD_list_dir + 'test_list.txt')]

        for v in range(len(video_list)):
            video = video_list[v]
            
            ref_frame_5, ref_name = utils.read_img(os.path.join(self.args.SDSD_ref_data_dir, video, '00005.png'))
            ref_frame_5 = utils.img2tensor(ref_frame_5).cuda() 
            
            frame_i0, frame_i0_filename = utils.read_img(os.path.join(self.args.SDSD_data_dir, video, "00002.png" ))
            frame_i1, frame_i1_filename = utils.read_img(os.path.join(self.args.SDSD_data_dir, video, "00003.png" ))
            frame_i2, frame_i2_filename = utils.read_img(os.path.join(self.args.SDSD_data_dir, video, "00004.png" ))
            frame_i3, frame_i3_filename = utils.read_img(os.path.join(self.args.SDSD_data_dir, video, "00005.png" ))
            frame_i4, frame_i4_filename = utils.read_img(os.path.join(self.args.SDSD_data_dir, video, "00006.png" ))
            frame_i5, frame_i5_filename = utils.read_img(os.path.join(self.args.SDSD_data_dir, video, "00007.png" ))
   
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
       
                ref_frame_5, f_h_pad, f_w_pad = align_to_64(ref_frame_5, 64) 
                [b, c, h, w] = frame_i0.shape
                
                frame_first_input  = torch.cat((frame_i0.detach(), frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach()), 1) 
                frame_second_input = torch.cat((frame_i1.detach(), frame_i2.detach(), frame_i3.detach(), frame_i4.detach(), frame_i5.detach()), 1) 
                
                val_first_frame_pred, val_second_frame_pred = self.Enhance_model(frame_first_input, frame_second_input) # [:,:,100:356, 400:656]  
                                
                # crop padding
                val_second_frame_pred = val_second_frame_pred[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                ref_frame_5 = ref_frame_5[:, :, 0:h-f_h_pad, 0:w-f_w_pad] 
                
                
                if self.args.use_tensorboard:
                    self.val_images = {}
                    self.val_images['Val/val_frame_pred'] = denorm(val_second_frame_pred.detach().cpu())
                    self.val_images['Val/Ref_frame'] = denorm(ref_frame_5.detach().cpu())
                    for tag, image in self.val_images.items():
                        self.logger.images_summary(tag, image, epoch)
                val_second_frame_pred = utils.tensor2img(val_second_frame_pred) 
                ref_frame_5 = utils.tensor2img(ref_frame_5)

            # Validation SSIM PSNR
            if self.args.is_test_psnr_ssim:
                SSIM_result = calc_ssim(val_second_frame_pred * 255.0, ref_frame_5 * 255.0)
                Total_SSIM += SSIM_result
                PSNR_result = calc_psnr(val_second_frame_pred * 255.0, ref_frame_5 * 255.0)
                Total_PSNR += PSNR_result

        if self.args.use_tensorboard:
            self.val_Metric = {} 
            ave_PSNR =  Total_PSNR / len(video_list)
            ave_SSIM =  Total_SSIM / len(video_list)
            self.val_Metric['Val/PSNR'] =  ave_PSNR
            self.val_Metric['Val/SSIM'] =  ave_SSIM
            print("Average PSNR :",ave_PSNR, "Average SSIM :",ave_SSIM)
            for tag, value in self.val_Metric.items():
                self.logger.scalar_summary(tag, value, epoch)

    # def learning_rate_decay(self, args, optimizer, epoch):
    #    """
    #    Sets the learning rate
    #    """
    #    current_epoch = epoch
    #    if current_epoch < 500:
    #        lr = self.args.lr_init
    #        for param_group in optimizer.param_groups:
    #            param_group['lr'] = lr
    #    else:
    #        lr = self.args.lr_init * 0.5
    #        for param_group in optimizer.param_groups:
    #            param_group['lr'] = lr
    #    return lr

    def train_syn(self):
        """ Train Low Light Enhancement Video ."""
        # start from scratch or trained models
        if self.args.pretrained_model:
            resume_epochs = findLastCheckpoint(self.model_save_path)
            self.load_pretrained_model(resume_epochs)
        else:
            resume_epochs = 0
        
        # define loss
        vgg_model = VGG19_Extractor(output_layer_list=[3,8,13,22]).cuda()
        criterion_ssim = pytorch_ssim.SSIM(window_size = 11)
        criterion_l1 = nn.L1Loss(size_average=True)
        criterion_l2 = nn.MSELoss(size_average=True)
        # MMD loss

        # start training
        print("======================================= start training =======================================")
        self.start_time = time.time()        
        while self.Enhance_model.epoch < self.args.total_epochs:
            self.Enhance_model.epoch += 1
            self.Raft_model.eval()

            ### learning rate update
            lr = self.learning_rate_decay(self.args, self.optimizer, self.Enhance_model.epoch)
            print("Learning rate (lr) basic model optimizer(3D and Fusion): [{}] ======".format(lr))
            
            for iteration, (batch_low, filename_low, batch_normal, filename_normal) in enumerate(self.loaders.ref, 1):
                assert filename_low[0][0].split("/")[-2] == filename_normal[0][0].split("/")[-2] # assert video name
                assert filename_low[0][0].split("/")[-1] == filename_normal[0][0].split("/")[-1] # assert frame name
              
                frame_low_i = []
                name_low_i = []
                frame_normal_i = []
                name_normal_i = []

                self.filename_low = filename_low
                self.filename_normal = filename_normal
                for t in range(self.args.sample_frames):
                    frame_low_i.append(batch_low[t].cuda())
                    frame_normal_i.append(batch_normal[t].cuda())
                    
                    name_low_i.append(filename_low[t])
                    name_normal_i.append(filename_normal[t])

                # Clear gradients
                self.optimizer.zero_grad()

                [b, c, h, w] = frame_low_i[0].shape
                # sample low frames
                frame_low_i0 = frame_low_i[0]
                frame_low_i1 = frame_low_i[1]
                frame_low_i2 = frame_low_i[2]
                frame_low_i3 = frame_low_i[3]            
                frame_low_i4 = frame_low_i[4] 
                frame_low_i5 = frame_low_i[5]
                
                # sample normal frames
                frame_normal_i0 = frame_normal_i[0]
                frame_normal_i1 = frame_normal_i[1]
                frame_normal_i2 = frame_normal_i[2]
                frame_normal_i3 = frame_normal_i[3]            
                frame_normal_i4 = frame_normal_i[4]
                frame_normal_i5 = frame_normal_i[5]
                
                flow_warping = Resample2d().cuda()
                
                # consecutive six frames
                frame_input = torch.cat((frame_low_i0.detach(), frame_low_i1.detach(), frame_low_i2.detach(), frame_low_i3.detach(), frame_low_i4.detach(), frame_low_i5.detach()), 1)    
                first_frame, second_frame = self.Enhance_model(frame_input) 
                
                # cycle self-regularization optical flow 
                _, flow_up_21 = self.Raft_model(second_frame, first_frame, iters=8, test_mode=True)
                warp_frame1 = flow_warping(first_frame, flow_up_21)                
                noc_mask1 = torch.exp(- 50 * torch.sum(second_frame - warp_frame1, dim=1).pow(2) ).unsqueeze(1)

                _, flow_up_12 = self.Raft_model(first_frame, second_frame, iters=8, test_mode=True)
                warp_frame2 = flow_warping(second_frame, flow_up_12)                
                noc_mask2 = torch.exp(- 50 * torch.sum(first_frame - warp_frame2, dim=1).pow(2) ).unsqueeze(1)
                                
                # Loss
                vgg_loss  = F_loss(first_frame, frame_normal_i2, vgg_model) + F_loss(second_frame, frame_normal_i3, vgg_model)
                ssim_loss = criterion_ssim(first_frame, frame_normal_i2) + criterion_ssim(second_frame, frame_normal_i3) 
                tw_loss   = criterion_l1(first_frame * noc_mask2, warp_frame2 * noc_mask2) + criterion_l1(second_frame * noc_mask1, warp_frame1 * noc_mask1) 
                overall_loss = vgg_loss + ssim_loss + 0.5 * tw_loss
                overall_loss.backward()
                self.overall_loss = overall_loss.item()
                self.vgg_loss = vgg_loss.item()
                self.ssim_loss = ssim_loss.item()
                self.tw_loss = tw_loss.item() * 10000.0
                
                # update
                self.optimizer.step()
                # visulization
                if self.args.use_tensorboard:
                    self.train_images = {}
                    self.train_images['Train/train_frame_pred_mid'] = denorm(first_frame.detach().cpu())
                    self.train_images['Train/Reference_frame_mid'] = denorm(frame_normal_i2.detach().cpu())
                    for tag, image in self.train_images.items():
                        self.logger.images_summary(tag, image, self.Enhance_model.epoch)

            ### print info and save models
            self.print_info(self.Enhance_model.epoch, self.args.total_epochs)
            ### using tensorboard for loss visulization    
            if self.args.use_tensorboard:
                self.loss = {}
                self.loss['Train/overall'] =  self.overall_loss
                self.loss['Train/vgg'] = self.vgg_loss
                self.loss['Train/ssim'] = self.ssim_loss
                self.loss['Train/tw'] = self.tw_loss
                for tag, value in self.loss.items():
                    self.logger.scalar_summary(tag, value, self.Enhance_model.epoch)
               
            ### model_validation
            self.model_validation(self.Enhance_model.epoch)
        print("======================================= Complete training =======================================")
    
    def train_SDSD(self):
        """ Train Low Light Enhancement Video ."""
        # start from scratch or trained models
        if self.args.pretrained_model:
            resume_epochs = findLastCheckpoint_SDSD(self.model_save_path)
            resume_epochs = 1000
            self.load_pretrained_model(resume_epochs)
        else:
            resume_epochs = 0
        
        # define loss
        vgg_model = VGG19_Extractor(output_layer_list=[3,8,13,22]).cuda()
        criterion_ssim = pytorch_ssim.SSIM(window_size = 11)
        criterion_l1 = nn.L1Loss(size_average=True)
        criterion_l2 = nn.MSELoss(size_average=True)

        # start training
        print("======================================= start training =======================================")
        self.start_time = time.time()        
        while self.Enhance_model.epoch < self.args.total_epochs:
            self.Enhance_model.epoch += 1
            self.Raft_model.eval()

            ### learning rate update
            lr = self.learning_rate_decay(self.args, self.optimizer, self.Enhance_model.epoch)
            print("Learning rate (lr) basic model optimizer(3D and Fusion): [{}] ======".format(lr))
            
            for iteration, (batch_low, filename_low, batch_normal, filename_normal) in enumerate(self.loaders.ref, 1):
                assert filename_low[0][0].split("/")[-2] == filename_normal[0][0].split("/")[-2] # assert video name
                assert filename_low[0][0].split("/")[-1] == filename_normal[0][0].split("/")[-1] # assert frame name
              
                frame_low_i = []
                name_low_i = []
                frame_normal_i = []
                name_normal_i = []

                self.filename_low = filename_low
                self.filename_normal = filename_normal
                for t in range(self.args.sample_frames):
                    frame_low_i.append(batch_low[t].cuda())
                    frame_normal_i.append(batch_normal[t].cuda())
                    
                    name_low_i.append(filename_low[t])
                    name_normal_i.append(filename_normal[t])

                # Clear gradients
                self.optimizer.zero_grad()

                [b, c, h, w] = frame_low_i[0].shape
                # sample low frames
                frame_low_i0 = frame_low_i[0]
                frame_low_i1 = frame_low_i[1]
                frame_low_i2 = frame_low_i[2]
                frame_low_i3 = frame_low_i[3]            
                frame_low_i4 = frame_low_i[4] 
                frame_low_i5 = frame_low_i[5]
                
                # sample normal frames
                frame_normal_i0 = frame_normal_i[0]
                frame_normal_i1 = frame_normal_i[1]
                frame_normal_i2 = frame_normal_i[2]
                frame_normal_i3 = frame_normal_i[3]            
                frame_normal_i4 = frame_normal_i[4]
                frame_normal_i5 = frame_normal_i[5]
                
                flow_warping = Resample2d().cuda()
                
                # the first 5 consecutive frames
                frame_first_input = torch.cat((frame_low_i0.detach(), frame_low_i1.detach(), frame_low_i2.detach(), frame_low_i3.detach(), frame_low_i4.detach()), 1)     
                frame_second_input = torch.cat((frame_low_i1.detach(), frame_low_i2.detach(), frame_low_i3.detach(), frame_low_i4.detach(), frame_low_i5.detach()), 1)
                
                first_frame, second_frame = self.Enhance_model(frame_first_input, frame_second_input) 
                
                # cycle self-regularization optical flow 
                _, flow_up_21 = self.Raft_model(second_frame, first_frame, iters=8, test_mode=True)
                warp_frame1 = flow_warping(first_frame, flow_up_21)                
                noc_mask1 = torch.exp(- 50 * torch.sum(second_frame - warp_frame1, dim=1).pow(2) ).unsqueeze(1)

                _, flow_up_12 = self.Raft_model(first_frame, second_frame, iters=8, test_mode=True)
                warp_frame2 = flow_warping(second_frame, flow_up_12)                
                noc_mask2 = torch.exp(- 50 * torch.sum(first_frame - warp_frame2, dim=1).pow(2) ).unsqueeze(1)
                               
                # Loss
                vgg_loss  = F_loss(first_frame, frame_normal_i2, vgg_model) + F_loss(second_frame, frame_normal_i3, vgg_model)
                ssim_loss = criterion_ssim(first_frame, frame_normal_i2) + criterion_ssim(second_frame, frame_normal_i3) 
                tw_loss   = criterion_l1(first_frame * noc_mask2, warp_frame2 * noc_mask2) + criterion_l1(second_frame * noc_mask1, warp_frame1 * noc_mask1) 
                overall_loss = vgg_loss + ssim_loss + 0.5 * tw_loss
                overall_loss.backward()
                self.overall_loss = overall_loss.item()
                self.vgg_loss = vgg_loss.item()
                self.ssim_loss = ssim_loss.item()
                self.tw_loss = tw_loss.item() * 10000.0
                
                # update
                self.optimizer.step()
                # visulization
                if self.args.use_tensorboard:
                    self.train_images = {}
                    self.train_images['Train/train_frame_pred_mid'] = denorm(first_frame.detach().cpu())
                    self.train_images['Train/Reference_frame_mid'] = denorm(frame_normal_i2.detach().cpu())
                    for tag, image in self.train_images.items():
                        self.logger.images_summary(tag, image, self.Enhance_model.epoch)

            ### print info and save models
            self.print_info(self.Enhance_model.epoch, self.args.total_epochs)
            ### using tensorboard for loss visulization    
            if self.args.use_tensorboard:
                self.loss = {}
                self.loss['Train/overall'] =  self.overall_loss
                self.loss['Train/vgg'] = self.vgg_loss
                self.loss['Train/ssim'] = self.ssim_loss
                self.loss['Train/tw'] = self.tw_loss
                for tag, value in self.loss.items():
                    self.logger.scalar_summary(tag, value, self.Enhance_model.epoch)
               
            ### model_validation
            self.model_validation_SDSD(self.Enhance_model.epoch)
        print("======================================= Complete training =======================================")