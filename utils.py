#-*-coding:utf-8-*-
from PIL import Image
import os, sys, random, math, cv2, pickle, subprocess
import torch
import torch.nn as nn
import math
import re
import glob
import numbers
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from losses import *
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import csv
import random
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import torchvision
import scipy.misc 
from torch.optim.optimizer import Optimizer, required
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


FLO_TAG = 202021.25
EPS = 1e-12

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


class Logger(object):
    """Create a tensorboard logger to log_dir."""
    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
    
    def images_summary(self, tag, images, step):
        """Log a list of images."""
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()

            scipy.misc.toimage(img).save(s, format="png")
            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
    
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


######################################################################################
##  Common utils
######################################################################################  

def findLastCheckpoint_SDSD(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'SDSD-Video_*.pth'))
   
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*SDSD-Video_(.*).pth.*", file_)
          
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0    
    return initial_epoch 


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'videoSRC-Video_*.pth'))
   
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*videoSRC-Video_(.*).pth.*", file_)
          
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch 


def align_to_64(frame_i0, divide):
    [b, c, h, w] = frame_i0.shape
    h_pad = int(np.floor(h/divide)+1)*divide
    w_pad = int(np.floor(w/divide)+1)*divide
    frame_i0_pad = F.pad(frame_i0, pad = [0, w_pad-w, 0, h_pad-h], mode='replicate')
    return frame_i0_pad, h_pad-h, w_pad-w

def create_folder(root_dir, path, version):
    if not os.path.exists(os.path.join(root_dir, path, version)):
        os.makedirs(os.path.join(root_dir, path, version))

def denorm(x):
    out = (x + 1) / 2.0
    out = out * 255.
    return out.clamp_(0, 255)

def str2bool(v):
    return v.lower() in ('true')

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def img2tensor(img):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))
    return img_t

def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

# save image in tester
def save_img(img, filename):
    print("Save %s" %filename)
    if img.ndim == 3:
        img = img[:, :, ::-1] ### RGB to BGR
    ## clip to [0, 1]
    img = np.clip(img, 0, 1)
    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)
    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


######################################################################################
##  Create dataloader
######################################################################################  
class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    def __len__(self):
        return len(self.indices)
    
def create_data_loader(data_set, args, mode):
    if mode == 'train':
        ### generate data sampler and loader
        indices = np.random.permutation(len(data_set))
        indices = np.tile(indices, args.repeat_times)  # 67 * 8
        sampler = SubsetSequentialSampler(indices) 
        data_loader = DataLoader(dataset=data_set, num_workers=args.threads, batch_size=args.train_batch_size, sampler=sampler, pin_memory=True, drop_last=True)
    else:
        data_loader = DataLoader(dataset=data_set, num_workers=1, batch_size=1, pin_memory=True, drop_last=False)
    return data_loader

######################################################################################
##  Flow utility
######################################################################################
def read_flo(filename):

    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' %filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            #print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))
    return flow

def save_flo(flow, filename):
    with open(filename, 'wb') as f:
        tag = np.array([FLO_TAG], dtype=np.float32)
        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        tag.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)
        
def flow_to_rgb(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))
    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.float32(img) / 255.0

######################################################################################
##  Image utility
######################################################################################

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
    return img, filename

def read_img_DRV(filename, grayscale=0):
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
    return img, filename
