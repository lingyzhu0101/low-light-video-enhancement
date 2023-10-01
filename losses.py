# -*-coding:utf-8-*-

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import numpy as np
from math import pi, exp


################################## CharbonnierLoss And EdgeLoss ############################################
# code from https://github.com/swz30/MPRNet/blob/main/Denoising/losses.py

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
    
        return loss
        
################################## MMD Loss ############################################
# code from https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0) # [16, 128, 64, 64]                 
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target): # [8, 128, 64, 64] [8, 128, 64, 64]
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

##################################SMOID Loss ############################################
# code from https://github.com/cao-cong/Seeing-Motion-in-the-Dark-pytorch/blob/master/vgg.py
class VGG19_Extractor(nn.Module):
    def __init__(self,output_layer_list=[]):
        super(VGG19_Extractor,self).__init__()

        self.vgg_features=models.vgg19(pretrained=True).features
        self.module_list=list(self.vgg_features.modules())[1:]
        self.output_layer_list=output_layer_list

        self.mean=nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1),requires_grad=False)
        self.std=nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1),requires_grad=False)


    def forward(self,x):
        x=x-self.mean.repeat(x.size(0),1,x.size(2),x.size(3))
        x=x/self.std.repeat(x.size(0),1,x.size(2),x.size(3))
        output_list=[]
        output_list.append(x)
        for module_idx, module in enumerate(self.module_list):
            x=module(x)
            if module_idx in self.output_layer_list:
                output_list.append(x)
        
        return output_list

def F_loss(img1, img2, vgg_model):
    output_1_vgg_0, output_1_vgg_1, output_1_vgg_2, output_1_vgg_3, output_1_vgg_4 = vgg_model(img1)
    output_2_vgg_0, output_2_vgg_1, output_2_vgg_2, output_2_vgg_3, output_2_vgg_4 = vgg_model(img2)
    loss_0 = reduce_mean(output_1_vgg_0, output_2_vgg_0)
    loss_1 = reduce_mean(output_1_vgg_1, output_2_vgg_1)
    loss_2 = reduce_mean(output_1_vgg_2, output_2_vgg_2)
    loss_3 = reduce_mean(output_1_vgg_3, output_2_vgg_3)
    loss_4 = reduce_mean(output_1_vgg_4, output_2_vgg_4)
    return loss_0 + loss_1 + loss_2 + loss_3 + loss_4
    
def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

#################################################################################

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class AngularLoss(torch.nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos_criterion = torch.nn.CosineSimilarity(dim=1)
        cos = cos_criterion(feature1, feature2)
        clip_bound = 0.999999
        cos = torch.clamp(cos, -clip_bound, clip_bound)
        if False:
            return 1 - torch.mean(cos)
        else:
            return torch.mean(torch.acos(cos)) * 180 / pi


class MultiscaleRecLoss(nn.Module):
    def __init__(self, scale=3, rec_loss_type='l1', multiscale=True):
        super(MultiscaleRecLoss, self).__init__()
        self.multiscale = multiscale
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(rec_loss_type))
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        if self.multiscale:
            self.weights = [1.0, 1.0/2, 1.0/4]
            self.weights = self.weights[:scale]

    def forward(self, input, target):
        loss = 0
        pred = input.clone()
        gt = target.clone()
        if self.multiscale:
            for i in range(len(self.weights)):
                loss += self.weights[i] * self.criterion(pred, gt)
                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            loss = self.criterion(pred, gt)
        return loss


def hingeloss(x, y, mode='fake'):
    if mode == 'fake':
        return torch.mean(nn.ReLU()(x + y))
    elif mode == 'real':
        return torch.mean(nn.ReLU()(x - y))
    else:
        raise NotImplementedError("=== Mode [{}] is not implemented. ===".format(mode))

def diff(x, y, mode=True):
    if mode:
        return x - torch.mean(y)
    else:
        return torch.mean(x) - y

def calc_l2(x, y, mode=False):
    if mode:
        return torch.mean((x - y) ** 2)
    else:
        return torch.mean((x + y) ** 2)


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'rahinge':
            pass
        elif gan_mode == 'rals':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)
    
    def loss(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        if self.gan_mode == 'original': # cross entropy loss
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(real_preds, target_tensor)
                return loss
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(fake_preds, target_tensor)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'ls':   
            if for_real:  
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                return F.mse_loss(real_preds, target_tensor)
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                return F.mse_loss(fake_preds, target_tensor)
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'hinge':
            if for_real:
                if for_discriminator:                
                    if target_is_real:
                        minval = torch.min(real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(real_preds)
                return loss
            elif for_fake:
                if for_discriminator:                
                    if target_is_real:
                        minval = torch.min(fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(fake_preds)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'rahinge':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))
                return loss / 2
                
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
                return loss / 2
        elif self.gan_mode == 'rals':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff - 1) ** 2) + torch.mean((f_r_diff + 1) ** 2)
                return loss / 2
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff + 1) ** 2) + torch.mean((f_r_diff - 1) ** 2)
                return loss / 2
        else:
            # wgan
            if for_real:
                if target_is_real:
                    return -real_preds.mean()
                else:
                    return real_preds.mean()
            elif for_fake:
                if target_is_real:
                    return -fake_preds.mean()
                else:
                    return fake_preds.mean()
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")

    def __call__(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        ## computing loss is a bit complicated because |input| may not be
        ## a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(real_preds, list):
            loss = 0
            for (pred_real_i, pred_fake_i) in zip(real_preds, fake_preds):
                if isinstance(pred_real_i, list):
                    pred_real_i = pred_real_i[-1]
                if isinstance(pred_fake_i, list):
                    pred_fake_i = pred_fake_i[-1]

                loss_tensor = self.loss(pred_real_i, pred_fake_i, target_is_real, for_real, for_fake, for_discriminator)

                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss
        else:
            return self.loss(real_preds, target_is_real, for_discriminator)


