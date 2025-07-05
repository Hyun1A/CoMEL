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

from main.single_default.abmil_trainer import ABMIL_Trainer


MIL_Trainer = {"attmil": ABMIL_Trainer}


def get_trainer(args):
    return MIL_Trainer[args.model]


def one_fold(args,k,train_p, train_l, test_p, test_l,val_p,val_l):
    # ---> Initialization
    seed_torch(args.seed)

    trainer = get_trainer(args)
    mil_trainer = trainer(args, k, train_p, train_l, test_p, test_l,val_p,val_l)
    mil_trainer.train()



def main():
    args = get_config()
            
    os.makedirs(os.path.join(args.model_path), exist_ok=True)

    args.pl_path = os.path.join(args.model_path, "pl")
    os.makedirs(args.pl_path, exist_ok=True)

    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    if args.model == 'clam_sb':
        args.cls_alpha= .7
        args.aux_alpha = .3
    elif args.model == 'clam_mb':
        args.cls_alpha= .7
        args.aux_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.aux_alpha = 0.5

    if args.datasets == 'cm16':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets == 'tcga':
        args.num_workers = 0
        args.always_test = True
    
    if args.wandb:
        wandb.init(project=args.project,name=args.title,config=args,dir=os.path.join(args.model_path))
        
    print(args)

    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)

    # set seed
    seed_torch(args.seed)

    # --->generate dataset
    args.dataset_root = args.dataset_root.replace("#organ#", args.organ)
    if args.datasets.lower() == 'cm16':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index].astype(int)

    elif args.datasets.lower() == 'paip':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index].astype(int)

    elif args.datasets.lower() == 'tcga':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index].astype(int)


    for k in range(args.fold_start, args.fold_end):
        if not args.no_log:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
            print(f'Dataset: {args.datasets}')
            print(f'Organ: {args.organ}')
            

        if args.cv_fold > 1:
            train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

        one_fold(args,k,train_p, train_l, test_p, test_l,val_p,val_l)



if __name__ == '__main__':
    main()
