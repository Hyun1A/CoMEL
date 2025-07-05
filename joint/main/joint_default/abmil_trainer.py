import os,sys
if "main" in sys.path[0]:
    sys.path[0] = "/".join(sys.path[0].split('/')[:-2])
sys.path = [sys.path[0]] + [f"{sys.path[0]}/ssl_engine"] + sys.path[1:]

from copy import deepcopy
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse
from configs import get_config
from modules import attmil,clam,dsmil,transmil,mean_max,rrt
import losses

from torch.nn.functional import one_hot

from torch.cuda.amp import GradScaler
from contextlib import suppress
import time

from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict
# from vis.vis_utils import visualize_prediction

from utils import *


from main.single_default.abmil_trainer import ABMIL_Trainer as ABMIL_Trainer_Single

class ABMIL_Trainer(ABMIL_Trainer_Single):
    def __init__(self, args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list):

        self.dataset_root_organ_list = dataset_root_organ_list

        super().__init__(args, k, train_p, train_l, test_p, test_l,val_p,val_l)

    def set_dataset(self, args, k):
        if args.datasets.lower() == 'cm16,paip':

            train_sets, test_sets, val_sets = [], [], []
            for d_idx, (train_p, train_l, test_p, test_l, val_p, val_l) in enumerate(zip(self.train_p,self.train_l,self.test_p,self.test_l,self.val_p,self.val_l)):

                dataset_root = self.dataset_root_organ_list[d_idx]

                train_set = C16Dataset(train_p[k],train_l[k],root=dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True, patch_labels=True, return_coords=True)
                
                test_set = C16Dataset(test_p[k],test_l[k],root=dataset_root,persistence=args.persistence,\
                                                    keep_same_psize=args.same_psize, patch_labels=True, return_coords=True)
                
                if args.val_ratio != 0.:
                    val_set = C16Dataset(val_p[k],val_l[k],root=dataset_root,persistence=args.persistence,\
                                                    keep_same_psize=args.same_psize, patch_labels=True, return_coords=True)
                else:
                    val_set = test_set

                train_sets.append(train_set)
                test_sets.append(test_set)
                val_sets.append(val_set)

            self.train_set = C16Dataset_Joint(train_sets) 
            self.test_set = C16Dataset_Joint(test_sets)
            self.val_set = C16Dataset_Joint(val_sets)


        elif args.datasets.lower() == 'tcga':
            self.train_set = TCGADataset(self.train_p[k],self.train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True,_type=args.tcga_sub)
            self.test_set = TCGADataset(self.test_p[k],self.test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
            if args.val_ratio != 0.:
                self.val_set = TCGADataset(self.val_p[k],self.val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
            else:
                self.val_set = self.test_set        
            
        return self.train_set, self.val_set, self.test_set,
            
