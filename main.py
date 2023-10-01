#-*-coding:utf-8-*-
import os
import argparse
import utils
from trainer import Trainer
from tester import Tester
from utils import create_folder, setup_seed
from config import get_config
import torch
from munch import Munch
import datasets_multiple

def main(args):
    setup_seed(args.seed)
    
    create_folder(args.save_root_dir, args.version, args.model_save_path)
    create_folder(args.save_root_dir, args.version, args.sample_path)
    create_folder(args.save_root_dir, args.version, args.log_path)
    create_folder(args.save_root_dir, args.version, args.val_SDSD)
    create_folder(args.save_root_dir, args.version, args.val_syn)
    create_folder(args.save_root_dir, args.version, args.test_SDSD)
    create_folder(args.save_root_dir, args.version, args.test_syn)
    
    if args.mode == 'train':
        dataset_combined_train = datasets_multiple.MultiFramesDataset_Combined(args, "train")
        get_train_loader_combined = utils.create_data_loader(dataset_combined_train, args, "train")
        
        loaders_combined = Munch(ref=get_train_loader_combined)
        trainer = Trainer(loaders_combined, args)
        trainer.train()
        
        
        # train sdsd
        #train_dataset_SDSD = datasets_multiple.MultiFramesDataset_SDSD(args, "train") 
        #train_data_loader_SDSD = utils.create_data_loader(train_dataset_SDSD, args, "train")
        #loaders_SDSD = Munch(ref=train_data_loader_SDSD)
        #trainer = Trainer(loaders_SDSD, args)
        #trainer.train_SDSD()
        
        # train synthesized data
        #train_dataset = datasets_multiple.MultiFramesDataset_syn(args, "train") 
        #train_data_loader = utils.create_data_loader(train_dataset, args, "train")
        #loaders = Munch(ref=train_data_loader)
        #trainer = Trainer(loaders, args)
        #trainer.train_syn()
        
    elif args.mode == 'test':
        tester = Tester(args)
        tester.test_SDSD()
        #tester.test_videoSRC()
    
    else:
        raise NotImplementedError('Mode [{}] is not found'.format(args.mode))


if __name__ == '__main__':
    args = get_config()
    main(args)