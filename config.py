#-*-coding:utf-8-*-

import argparse
from utils import str2bool

def get_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--mode',                 type=str,               default='train',           help='train|test')
    parser.add_argument('--adv_loss_type',        type=str,               default='rahinge',         help='adversarial Loss: ls|original|hinge|rahinge|rals')
    parser.add_argument('--version',              type=str,               default='Enhance-Video',   help='UEGAN')
    parser.add_argument('--init_type',            type=str,               default='xavier',          help='normal|xavier|kaiming|orthogonal')
    parser.add_argument('--parallel',             type=str2bool,          default=False,             help='use multi-GPU for training')
    parser.add_argument('--gpu_ids',                                      default=[3,2])
    parser.add_argument('--use_tensorboard',      type=str,               default=True)
    parser.add_argument('--is_test_psnr_ssim',    type=str2bool,          default=True)
    parser.add_argument('--is_test_nima',         type=str2bool,          default=False)
    parser.add_argument('--is_print_network',     type=str2bool,          default=True)
    parser.add_argument('--pretrained_model',     type=int,               default=0,                 help='pretrained model eppoch')
    parser.add_argument('--total_epochs',         type=int,               default=300,               help='total epochs to update the generator')
    parser.add_argument('--train_batch_size',     type=int,               default=4,                 help='mini batch size') # 32
    parser.add_argument('--repeat_times',         type=int,               default=10,                help='repeat times for dataloader') # 32
    parser.add_argument('--threads',              type=int,               default=8,                 help='subprocesses to use for data loading')
    parser.add_argument('--seed',                 type=int,               default=19960101,          help='Seed for random number generator')
    
    # create folder 
    parser.add_argument('--save_root_dir',        type=str,               default='./results')
    parser.add_argument('--model_save_path',      type=str,               default='models')
    parser.add_argument('--sample_path',          type=str,               default='sample')
    parser.add_argument('--log_path',             type=str,               default='logs')
    parser.add_argument('--val_SDSD',             type=str,               default='validation_SDSD')
    parser.add_argument('--val_syn',              type=str,               default='validation_syn')
    parser.add_argument('--test_SDSD',            type=str,               default='test_SDSD')
    parser.add_argument('--test_syn',             type=str,               default='test_syn')
    
    # model generator settings
    parser.add_argument('--input_channels',       type=int,               default=3,                 help='channel of input')
    parser.add_argument('--base_dim',             type=int,               default=32,                help='base feature dimension')
    parser.add_argument('--res_block_num',       type=int,                default=12,                help='number of residul block')
    # model discriminator settings 
    parser.add_argument('--d_conv_dim',           type=int,               default=32,                help='number of conv filters in the first layer of D')
    parser.add_argument('--d_use_sn',             type=str2bool,          default=True,              help='whether use spectral normalization in D')
    parser.add_argument('--d_act_fun',            type=str,               default='LeakyReLU',       help='LeakyReLU|ReLU|Swish|SELU|none')
    parser.add_argument('--d_norm_fun',           type=str,               default='none',            help='BatchNorm|InstanceNorm|none')
    
    # step size
    parser.add_argument('--log_step',             type=int,               default=1,                 help='print log step')
    parser.add_argument('--info_step',            type=int,               default=1,                 help='print loss step')
    parser.add_argument('--sample_step',          type=int,               default=250,               help='print sample image step')
    parser.add_argument('--model_save_epoch',     type=int,               default=1,                 help='set epoch to save model') 
    
    # dataset options
    parser.add_argument('--scale_min',            type=float,             default=0.5,               help='min scaling factor')
    parser.add_argument('--scale_max',            type=float,             default=2.0,               help='max scaling factor')
    parser.add_argument('--geometry_aug',         type=int,               default=1,                 help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('--crop_size',            type=int,               default=256,               help='crop patch to train model')
    parser.add_argument('--sample_frames',        type=int,               default=6,                 help='#frames for training')
    # SDSD Directories.
    parser.add_argument('--SDSD_list_dir',        type=str,               default='./data/SDSD_CUHK/')
    parser.add_argument('--SDSD_dir',             type=str,               default='./data/SDSD_CUHK/videoSDSD_low') 
    parser.add_argument('--SDSD_ref_dir',         type=str,               default='./data/SDSD_CUHK/videoSDSD_normal')
    # videoSRC Directories.
    parser.add_argument('--syn_list_dir',         type=str,               default='./data/DS_LOL/')
    parser.add_argument('--syn_dir',              type=str,               default='./data/DS_LOL/low_and_noise') 
    parser.add_argument('--syn_ref_dir',          type=str,               default='./data/DS_LOL/normal')
    # DRV Directories.
    parser.add_argument('--DRV_list_dir',         type=str,               default='./data/DRV_HKUST/')
    parser.add_argument('--DRV_dir',              type=str,               default='./data/DRV_HKUST/VBM4D_rawRGB') 
    parser.add_argument('--DRV_ref_dir',          type=str,               default='./data/DRV_HKUST/long')
    
    # learning rate and optimizer settings
    parser.add_argument('--lr_decay',             type=str2bool,          default=True,              help='learning decay settings')
    parser.add_argument('--lr_num_epochs_decay',  type=int,               default=150,               help='LambdaLR: epoch at starting learning rate')
    parser.add_argument('--lr_decay_ratio',       type=int,               default=150,               help='LambdaLR: ratio of linearly decay learning rate to zero')
    parser.add_argument('--lr_init',              type=float,             default=1e-4,              help='initial learning Rate')
    #parser.add_argument('--lr_offset',           type=int,               default=20,                help='epoch to start learning rate drop [-1 = no drop]')
    #parser.add_argument('--lr_step',             type=int,               default=50,                help='step size (epoch) to drop learning rate')
    #parser.add_argument('--lr_drop',             type=float,             default=0.5,               help='learning rate drop ratio')
    #parser.add_argument('--lr_min',              type=float,             default=0.01,              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    parser.add_argument('--optimizer_type',       type=str,               default='adam',            help='adam|rmsprop|SGD')
    parser.add_argument('--beta1',                type=float,             default=0.9,               help='beta1 for Adam optimizer')
    parser.add_argument('--beta2',                type=float,             default=0.999,             help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay',         type=float,             default=0.0001,            help='weight decay Adam optimizer')
    parser.add_argument('--alpha',                type=float,             default=0.9,               help='alpha for rmsprop optimizer')
    
    # raft model hyperparameter
    parser.add_argument('--use_pretrained_raft',  type=str2bool,          default=True,              help='use pretrained raft flow model')
    parser.add_argument('--raft_model_path',      default='Pretrained_RAFT/raft-things.pth',         help='use pretrained raft flow model')
    parser.add_argument('--small',                action='store_true',    help='use small model')
    parser.add_argument('--mixed_precision',      action='store_true',    help='use mixed precision')
    parser.add_argument('--iters',                type=int,               default=12)
    parser.add_argument('--wdecay',               type=float,             default=0.00005)
    parser.add_argument('--epsilon',              type=float,             default=1e-8)
    parser.add_argument('--dropout',              type=float,             default=0.0)
    parser.add_argument('--raft_lr',              type=float,             default=0.00002)
    return parser.parse_args()