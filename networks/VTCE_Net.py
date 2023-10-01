import functools
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.autograd import Variable
#from .ConvLSTM import ConvLSTM

#from ConvLSTM import ConvLSTM


################################################################### ConvLSTM Module #########################################################################################
### modified from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size).to(input_.device),
                torch.zeros(state_size).to(input_.device)
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


################################################################### ASPP Module #########################################################################################
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False) 

class ASPP(nn.Module):
    def __init__(self, in_channels=128, atrous_rates=[12, 24, 36]):
        super(ASPP, self).__init__()
        out_channels = 128
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



################################################################### Encoder Module #########################################################################################
def To3D(E1, group):
    [b, c, h, w] = E1.shape
    nf = int(c/group)
    E_list = []
    for i in range(0, group):
        tmp = E1[:, nf*i:nf*(i+1), :, :]
        tmp = tmp.view(b, nf, 1, h, w)
        E_list.append(tmp)
    E1_3d = torch.cat(E_list, 2)
    return E1_3d

def To2D(E1_3d):
    [b, c, g, h, w] = E1_3d.shape
    E_list = []
    for i in range(0, g):
        tmp = E1_3d[:, :, i, :, :]
        tmp = tmp.view(b, c, h, w)
        E_list.append(tmp)
    E1 = torch.cat(E_list, 1)
    return E1

class Blocks(nn.Module):
    def __init__(self,res_block_num,conv_channels):
        super(Blocks,self).__init__()
        self.res_block_num=res_block_num
        for res_block_idx in range(self.res_block_num):
            conv_layer_1=nn.Conv2d(conv_channels,conv_channels,kernel_size=3,padding=1,stride=1)
            conv_layer_2=nn.Conv2d(conv_channels,conv_channels,kernel_size=3,padding=1,stride=1)
            self.add_module('%d'%(2*res_block_idx),nn.Sequential(conv_layer_1))
            self.add_module('%d'%(2*res_block_idx+1),nn.Sequential(conv_layer_2))

    def __getitem__(self, index):
        if index < 0 or index >= len(self._modules):
            raise IndexError('index %d is out of range'%(index))

        return(self._modules[str(index)])

    def __len__(self):
        return self.res_block_num 


class Encoder_3D(nn.Module):
    def __init__(self,input_channels=3,base_dim=32, res_block_num=16):
        super(Encoder_3D,self).__init__()
        self.res_block_num = res_block_num
        # Encoder Net
        self.conv_1_1=nn.Conv2d(15,base_dim*5,kernel_size=3,padding=1,stride=1,groups=5)
        self.conv_1_2=nn.Conv2d(base_dim*5,base_dim*5,kernel_size=3,padding=1,stride=1,groups=5)
        self.conv_2_1=nn.Conv3d(base_dim,base_dim*2,kernel_size=(2, 3, 3),padding=(0,1,1),stride=(1,2,2))
        self.conv_2_2=nn.Conv2d(base_dim*2*4,base_dim*2*4,kernel_size=3,padding=1,stride=1,groups=4)
        self.conv_3_1=nn.Conv3d(base_dim*2,base_dim*2, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1))
        self.conv_3_2=nn.Conv2d(base_dim*2*3,base_dim*2*3,kernel_size=3,padding=1,stride=1,groups=3)
        self.conv_4_1=nn.Conv3d(base_dim*2,base_dim*4, kernel_size=(2, 3, 3), stride=(1,2,2), padding=(0,1,1))
        self.conv_4_2=nn.Conv2d(base_dim*4*2,base_dim*4*2,kernel_size=3,padding=1,stride=1,groups=2)
        self.conv_5_1=nn.Conv3d(base_dim*4, base_dim*4, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1))
        self.conv_5_2=nn.Conv2d(base_dim*4*1,base_dim*4*1,kernel_size=3,padding=1,stride=1,groups=1)

        # high-dim features
        self.res_block_list=Blocks(self.res_block_num,base_dim*4)
        self.convlstm = ConvLSTM(input_size=base_dim * 4, hidden_size = base_dim * 4, kernel_size=3)
        self.aspp = ASPP(in_channels=128, atrous_rates=[12, 24, 36])
        self.conv_after_res_block=nn.Conv2d(base_dim*4,base_dim*4,kernel_size=3,padding=1,stride=1)
       
    def forward(self, input_tensor, prev_state=None):
        self.lrelu=nn.LeakyReLU(negative_slope=0.2)
        conv_1 = self.lrelu(self.conv_1_1(input_tensor)) # [4, 160, 256, 256]
        conv_1 = self.conv_1_2(conv_1) # [4, 160, 256, 256]
        self.ln1 = nn.LayerNorm(conv_1.size()[1:], elementwise_affine=False)
        conv_1_cat = self.lrelu(self.ln1(conv_1)) 
        conv_1 = To3D(conv_1_cat, 5) # [4, 32, 5, 256, 256]
        
        conv_2 = self.conv_2_1(conv_1) # [4, 64, 4, 128, 128]  ### 3D Conv
        conv_2 = To2D(conv_2) #[4, 256, 128, 128]
        self.ln2 = nn.LayerNorm(conv_2.size()[1:], elementwise_affine=False)
        conv_2 = self.lrelu(self.ln2(self.conv_2_2(conv_2))) # [4, 256, 128, 128]
        conv_2 = To3D(conv_2, 4) # [4, 64, 4, 128, 128]

        conv_3 = self.conv_3_1(conv_2) # [4, 64, 3, 128, 128]  ### 3D Conv
        conv_3 = To2D(conv_3) # [4, 192, 128, 128]
        self.ln2 = nn.LayerNorm(conv_3.size()[1:], elementwise_affine=False)
        conv_3_cat = self.lrelu(self.ln2(self.conv_3_2(conv_3))) # [4, 192, 128, 128]
        conv_3 = To3D(conv_3_cat, 3) # [4, 64, 3, 128, 128]

        conv_4 = self.conv_4_1(conv_3) # [4, 128, 2, 64, 64]  ### 3D Conv
        conv_4 = To2D(conv_4) # [4, 256, 64, 64]
        self.ln2 = nn.LayerNorm(conv_4.size()[1:], elementwise_affine=False)
        conv_4 = self.lrelu(self.ln2(self.conv_4_2(conv_4))) # [4, 256, 64, 64]
        conv_4 = To3D(conv_4, 2) # [4, 128, 2, 64, 64]

        conv_5 = self.conv_5_1(conv_4) # [4, 128, 1, 64, 64]   ### 3D Conv
        conv_5 = To2D(conv_5) # [4, 128, 64, 64]
        self.ln2 = nn.LayerNorm(conv_5.size()[1:], elementwise_affine=False)
        conv_5 = self.lrelu(self.ln2(self.conv_5_2(conv_5))) # [4, 256, 64, 64]
      
        #res block
        conv_feature_end=conv_5 # [4, 128, 64, 64]
        for res_block_idx in range(self.res_block_num):
            conv_feature_begin = conv_feature_end
            conv_feature=self.res_block_list[2*res_block_idx](conv_feature_begin)
            self.ln_1 = nn.LayerNorm(conv_feature.size()[1:], elementwise_affine=False)
            conv_feature = self.res_block_list[2*res_block_idx+1](self.lrelu(self.ln_1(conv_feature)))
            self.ln_2 = nn.LayerNorm(conv_feature.size()[1:], elementwise_affine=False)
            conv_feature = self.ln_2(conv_feature)
            conv_feature_end = conv_feature_begin+conv_feature
      
        state = self.convlstm(conv_feature_end, prev_state)
        conv_feature_end = conv_feature_end + state[0] # [4, 128, 64, 64] + [4, 128, 64, 64] --> [4, 128, 64, 64]
        
        conv_feature_end =  self.aspp(conv_feature_end)
        conv_feature=self.conv_after_res_block(conv_feature_end) # [4, 128, 64, 64]
        self.ln = nn.LayerNorm(conv_feature.size()[1:], elementwise_affine=False)
        conv_feature = self.ln(conv_feature)
        conv_feature_end=conv_feature+conv_5 # [4, 128, 64, 64]            
        return conv_feature_end, conv_3_cat, conv_1_cat


################################################################### Decoder Module #########################################################################################
def deconv_2d(in_channels,out_channels):
    deconv=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    return nn.Sequential(*[deconv])

class Decoder_2D(nn.Module):
    def __init__(self,base_dim=32):
        super(Decoder_2D,self).__init__()
        # Decoder Net
        self.deconv_6=deconv_2d(base_dim*4,base_dim*2)
        self.conv_6_1=nn.Conv2d(base_dim*4,base_dim*2,kernel_size=3,padding=1,stride=1)
        self.conv_6_2=nn.Conv2d(base_dim*2,base_dim*2,kernel_size=3,padding=1,stride=1)
        self.deconv_7=deconv_2d(base_dim*2,base_dim*1)
        self.conv_7_1=nn.Conv2d(base_dim*2,base_dim*1,kernel_size=3,padding=1,stride=1)
        self.conv_7_2=nn.Conv2d(base_dim*1,base_dim*1,kernel_size=3,padding=1,stride=1)
        self.conv_8=nn.Conv2d(base_dim,3,kernel_size=3,padding=1,stride=1)
        
    def forward(self, input_tensor, conv_3_feature, conv_1_feature):
        self.lrelu=nn.LeakyReLU(negative_slope=0.2)
        conv_6_up=self.deconv_6(input_tensor)# [4, 64, 128, 128]  
        conv_6=torch.cat((conv_6_up,conv_3_feature[:,64:128,:,:]),dim=1) # [4, 64, 128, 128], conv_3_cat [4, 192, 128, 128] -- > [4, 128, 128, 128]
        conv_6=self.conv_6_1(conv_6) # [4, 64, 128, 128]
        self.ln6_1 = nn.LayerNorm(conv_6.size()[1:], elementwise_affine=False)
        conv_6=self.conv_6_2(self.lrelu(self.ln6_1(conv_6))) # [4, 64, 128, 128]
        self.ln6_2 = nn.LayerNorm(conv_6.size()[1:], elementwise_affine=False)
        conv_6=self.lrelu(self.ln6_2(conv_6)) # [4, 64, 128, 128]

        conv_7_up=self.deconv_7(conv_6) # [4, 32, 256, 256]
        conv_7=torch.cat((conv_7_up,conv_1_feature[:,64:96,:,:]),dim=1) # [4, 64, 256, 256]
        conv_7=self.conv_7_1(conv_7) # [4, 32, 256, 256]
        self.ln7_1 = nn.LayerNorm(conv_7.size()[1:], elementwise_affine=False)
        conv_7=self.conv_7_2(self.lrelu(self.ln7_1(conv_7))) # [4, 32, 256, 256]]
        self.ln7_2 = nn.LayerNorm(conv_7.size()[1:], elementwise_affine=False)
        conv_7=self.lrelu(self.ln7_2(conv_7))
       
        out=self.conv_8(conv_7)
        return out

################################################################### Total Module #########################################################################################
class VTCE_Net(nn.Module):
    def __init__(self,input_channels=3,base_dim=32,res_block_num=16):
        super(VTCE_Net, self).__init__()
        self.encoder = Encoder_3D(input_channels, base_dim, res_block_num)
        self.co_attention = AttentionGate_Wavelet(all_channel=128, all_dim=32 * 32)
        self.decoder = Decoder_2D(base_dim=32)
        
    def forward(self, input):
        # Encoder [4, 128, 64, 64] [4, 192, 128, 128] [4, 160, 256, 256]
        exemplar_latent_feature, exemplar_conv_3_cat, exemplar_conv_1_cat = self.encoder(input[:, 0:15, :, :]) 
        query_latent_feature, query_conv_3_cat, query_conv_1_cat = self.encoder(input[:, 3:18, :, :])
       
        x1, x2 = self.co_attention(exemplar_latent_feature, query_latent_feature) # [4, 128, 64, 64]   [4, 128, 64, 64]
        
        # Decoder
        frame_x1_out = self.decoder(x1, exemplar_conv_3_cat, exemplar_conv_1_cat) 
        frame_x2_out = self.decoder(x2, query_conv_3_cat, query_conv_1_cat)
       
        return frame_x1_out, frame_x2_out
        
###########################################################################################################################################################################

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2 # [4, 128, 32, 64]
    x02 = x[:, :, 1::2, :] / 2 # [4, 128, 32, 64]
    
    x1 = x01[:, :, :, 0::2] # [4, 128, 32, 32]
    x2 = x02[:, :, :, 0::2] # [4, 128, 32, 32]
    x3 = x01[:, :, :, 1::2] # [4, 128, 32, 32]
    x4 = x02[:, :, :, 1::2] # [4, 128, 32, 32]
    
    x_LL = x1 + x2 + x3 + x4  # [4, 128, 32, 32]
    x_HL = -x1 - x2 + x3 + x4 # [4, 128, 32, 32]
    x_LH = -x1 + x2 - x3 + x4 # [4, 128, 32, 32]
    x_HH = x1 - x2 - x3 + x4  # [4, 128, 32, 32]

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1) # [4, 512, 32, 32]


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size() # [4, 512, 32, 32]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width # [4 128 64 64]
    x1 = x[:, 0:out_channel, :, :] / 2                   # [4, 128, 32, 32]
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2     # [4, 128, 32, 32]
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2 # [4, 128, 32, 32]
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2 # [4, 128, 32, 32]

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h # [4, 128, 64, 64]



class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x): # [4, 128, 64, 64]
        return dwt_init(x) # [4, 512, 32, 32]


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x): # [4, 512, 32, 32]
        return iwt_init(x) # [4, 128, 64, 64]



class AttentionGate_Wavelet(nn.Module):  # spatial and channel attention module
    def __init__(self, all_channel=128, all_dim=32 * 32):  # 473./8=60 416./8=52
        super(AttentionGate_Wavelet, self).__init__()  
        
        self.DWT = DWT().cuda()
        self.IWT = IWT().cuda()
        
        self.channel = all_channel
        self.dim = all_dim
        # transpose 
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        
        # spatial gate
        self.gate1 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate2 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=128*2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=128)
        self.fc3 = nn.Linear(in_features=128*2, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=128)
        
        # frequency gate
        self.gate1_fre = nn.Conv2d(all_channel * 8, 1, kernel_size=1, bias=False)
        self.gate2_fre = nn.Conv2d(all_channel * 8, 1, kernel_size=1, bias=False)
        self.gate_s_fre = nn.Sigmoid()
        self.fc1_fre = nn.Linear(in_features=128*8, out_features=16)
        self.fc2_fre = nn.Linear(in_features=16, out_features=128*4)
        self.fc3_fre = nn.Linear(in_features=128*8, out_features=16)
        self.fc4_fre = nn.Linear(in_features=16, out_features=128*4)
        
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # conv after concat features
        self.conv1 = nn.Conv2d(all_channel * 3, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 3, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    

    def forward(self, frame1, frame2): # [4, 128, 64, 64] [4, 128, 64, 64]
        B, C, H, W = frame1.size()
        frame1 = F.interpolate(frame1, size=[int(H/2), int(W/2)], mode='bilinear', align_corners=False)
        frame2 = F.interpolate(frame2, size=[int(H/2), int(W/2)], mode='bilinear', align_corners=False)        
        fea_size = [int(H/2), int(W/2)]  # [4, 128, 32, 32] [4, 128, 32, 32]
        all_dim = int(H/2) * int(W/2)
        
        # reshape, transpose and activation
        frame1_flat = frame1.view(-1, frame2.size()[1], all_dim)  # N,C,H*W  [4, 128, 1024]
        frame2_flat = frame2.view(-1, frame2.size()[1], all_dim)  # N,C,H*W  [4, 128, 1024]
        frame1_t = torch.transpose(frame1_flat, 1, 2).contiguous()  # batch size x dim x num [4, 1024, 128]
        frame1_corr = self.linear_e(frame1_t)  # [4, 1024, 128]
        # common feature and normalization 
        A = torch.bmm(frame1_corr, frame2_flat) # [4, 1024, 1024]
        A1 = F.softmax(A.clone(), dim=1)  # [4, 1024, 1024] 
        B = F.softmax(torch.transpose(A, 1, 2), dim=1) # [4, 1024, 1024]
        # change mutual information, produce fake ones
        frame2_att = torch.bmm(frame1_flat, A1).contiguous() # [4, 128, 1024]
        frame1_att = torch.bmm(frame2_flat, B).contiguous() # [4, 128, 1024]
        
        # reshape the fake ones 
        input1_att = frame1_att.view(-1, frame2.size()[1], fea_size[0], fea_size[1]) # [4, 128, 32, 32]
        input2_att = frame2_att.view(-1, frame2.size()[1], fea_size[0], fea_size[1]) # [4, 128, 32, 32]
        input1_att_fre = input1_att.clone() # [4, 128, 32, 32]
        input2_att_fre = input2_att.clone() # [4, 128, 32, 32]
        
        ########### spatial branch ################## 
        # spacial attention and normalization 
        input1_mask = self.gate1(torch.cat([input1_att, input2_att], dim=1)) # [4, 1, 32, 32]
        input2_mask = self.gate2(torch.cat([input1_att, input2_att], dim=1))
        input1_mask = self.gate_s(input1_mask) # [4, 1, 32, 32]
        input2_mask = self.gate_s(input2_mask)
        
        # channel attention
        out_e = torch.cat([input1_att, input2_att], dim=1)
        out_e = out_e.mean(-1).mean(-1)
        out_e = out_e.view(out_e.size(0), -1)# [4, 256]
        out_e = self.fc1(out_e)# [4, 16]
        out_e = self.relu(out_e) # [4, 16]
        out_e = self.fc2(out_e) # [4, 128]
        out_e = self.sigmoid(out_e)# [4, 128]
        out_e = out_e.view(out_e.size(0), out_e.size(1), 1, 1) # [4, 128, 1, 1]   
        
        out_q = torch.cat([input1_att, input2_att], dim=1)
        out_q = out_q.mean(-1).mean(-1)
        out_q = out_q.view(out_q.size(0), -1)
        out_q = self.fc3(out_q)
        out_q = self.relu(out_q)
        out_q = self.fc4(out_q)
        out_q = self.sigmoid(out_q)
        out_q = out_q.view(out_q.size(0), out_q.size(1), 1, 1)
                
        # apply dual attention masks
        input1_att_spa = input1_att * input1_mask # [4, 128, 32, 32] 
        input2_att_spa = input2_att * input2_mask
        input2_att_spa = out_e * input2_att_spa
        input1_att_spa = out_q * input1_att_spa
        
        
        ########### frequency branch ################## 
        # spatial attention
        input1_att_fre = self.DWT(input1_att_fre) # [4, 512, 16, 16]
        input2_att_fre = self.DWT(input2_att_fre) # [4, 512, 16, 16]
               
        input1_mask_fre = self.gate1_fre(torch.cat([input1_att_fre, input2_att_fre], dim=1)) # [4, 1, 16, 16]
        input2_mask_fre = self.gate2_fre(torch.cat([input1_att_fre, input2_att_fre], dim=1))
        
        input1_mask_fre = self.gate_s_fre(input1_mask_fre) # [4, 1, 16, 16]
        input2_mask_fre = self.gate_s_fre(input2_mask_fre) # [4, 1, 16, 16]
        
        # channel attention
        out_e_fre = torch.cat([input1_att_fre, input2_att_fre], dim=1) # [4, 1024, 16, 16]
        out_e_fre = out_e_fre.mean(-1).mean(-1) # [4, 1024]
        out_e_fre = out_e_fre.view(out_e_fre.size(0), -1) # [4, 1024]
        out_e_fre = self.fc1_fre(out_e_fre) # [4, 16]
        out_e_fre = self.relu(out_e_fre) # [4, 16]
        out_e_fre = self.fc2_fre(out_e_fre) # [4, 512]
        out_e_fre = self.sigmoid(out_e_fre)# [4, 512]
        out_e_fre = out_e_fre.view(out_e_fre.size(0), out_e_fre.size(1), 1, 1) # [4, 512, 1, 1]   

        
        out_q_fre = torch.cat([input1_att_fre, input2_att_fre], dim=1) # [4, 1024, 16, 16]
        out_q_fre = out_q_fre.mean(-1).mean(-1) # [4, 1024]
        out_q_fre = out_q_fre.view(out_q_fre.size(0), -1) # [4, 1024]
        out_q_fre = self.fc3_fre(out_q_fre) # [4, 16]
        out_q_fre = self.relu(out_q_fre) # [4, 16]
        out_q_fre = self.fc4_fre(out_q_fre) # [4, 512]
        out_q_fre = self.sigmoid(out_q_fre) # [4, 512]
        out_q_fre = out_q_fre.view(out_q_fre.size(0), out_q_fre.size(1), 1, 1)
        
        # apply dual attention masks
        input1_att_fre = input1_att_fre * input1_mask_fre # [4, 512, 16, 16]
        input2_att_fre = input2_att_fre * input2_mask_fre
        input2_att_fre = out_e_fre * input2_att_fre # ([4, 512, 16, 16]
        input1_att_fre = out_q_fre * input1_att_fre
        
        input1_att_fre = self.IWT(input1_att_fre) # [4, 128, 32, 32]
        input2_att_fre = self.IWT(input2_att_fre)        
        

        # concate original feature
        input1_att = torch.cat([input1_att_spa, input1_att_fre, frame1], 1) # [4, 256, 32, 32]
        input2_att = torch.cat([input2_att_spa, input2_att_fre, frame2], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = self.bn1(input1_att)
        input2_att = self.bn2(input2_att)
        input1_att = self.prelu(input1_att)
        input2_att = self.prelu(input2_att)
        
        input1_att = F.interpolate(input1_att, size=[H, W], mode='bilinear', align_corners=False) # [4, 256, 64, 64]
        input2_att = F.interpolate(input2_att, size=[H, W], mode='bilinear', align_corners=False) # [4, 256, 64, 64]
       
        return input1_att, input2_att




if __name__== '__main__':
    from thop import profile
    #net = VTCE_Net().cuda()
    #print(net)
    #input = torch.randn(4, 18, 256, 256).cuda()
    #flops, params = profile(net, inputs=(input,))
    #total = sum([param.nelement() for param in net.parameters()])
    
    
    net = AttentionGate_Wavelet(all_channel=128, all_dim=32 * 32).cuda() # 0.62M # 2.49GFLOPs
    print(net)
    input = torch.randn(4, 128, 64, 64).cuda()
    flops, params = profile(net, inputs=(input, input,))
    total = sum([param.nelement() for param in net.parameters()])
    
    print('   Number of params: %.2fM' % (total / 1e6)) 
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9)) 
    
