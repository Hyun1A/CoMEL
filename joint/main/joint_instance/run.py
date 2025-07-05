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

from main.joint_instance.abmil_trainer import ABMIL_Trainer
from main.joint_instance.dsmil_trainer import DSMIL_Trainer 
from main.joint_instance.transmil_trainer import TransMIL_Trainer 
from main.joint_instance.rrtmil_trainer import RRTMIL_Trainer 
from main.joint_instance.amil_trainer_no_inst_cls import AMIL_Trainer 
from main.joint_instance.cdatmil_trainer import CDATMIL_Trainer
from main.joint_instance.cdatmil_ppl_trainer import CDATMIL_PPL_Trainer


#, TransMIL_Trainer, RRTMIL_Trainer, AMIL_Trainer

MIL_Trainer = {"attmil": ABMIL_Trainer, 
               "dsmil": DSMIL_Trainer,
               "transmil": TransMIL_Trainer,
               "rrtmil": RRTMIL_Trainer,
               "agentmil": AMIL_Trainer,
               "cdatmil": CDATMIL_Trainer,
               "cdatmil_ppl": CDATMIL_PPL_Trainer}


def get_trainer(method_name):
    return MIL_Trainer[method_name]


def one_fold(args,k,train_p_list, train_l_list, test_p_list, test_l_list,val_p_list,val_l_list,dataset_root_organ_list):
    # ---> Initialization
    seed_torch(args.seed)

    ppl = "_ppl" if args.use_ppl else ""
    method_name = f"{args.model}{ppl}"

    trainer = get_trainer(method_name)
    mil_trainer = trainer(args, k,train_p_list, train_l_list, test_p_list, test_l_list,val_p_list,val_l_list,dataset_root_organ_list)
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


    for k in range(args.fold_start, args.fold_end):
        if not args.no_log:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
            print(f'Dataset: {args.datasets}')
            print(f'Organ: {args.organ}')
            


        dataset_name_list = []
        dataset_organ_list = []
        dataset_root_organ_list = []

        train_p_list, train_l_list, test_p_list, test_l_list, val_p_list, val_l_list = [], [], [], [], [], []

        for d_idx, organ in enumerate(args.organ.split(",")):   
            dataset_name = "cm16" if organ=="mixed" else "paip"

            dataset_root_organ = args.dataset_root.replace("#dataset#", dataset_name).replace("#organ#", organ)

            dataset_name_list.append(dataset_name)
            dataset_organ_list.append(organ)
            dataset_root_organ_list.append(dataset_root_organ)


            if dataset_name.lower() == 'cm16':
                label_path=os.path.join(dataset_root_organ,'label.csv')
                p, l = get_patient_label(label_path)
                index = [i for i in range(len(p))]
                random.shuffle(index)
                p = p[index]
                l = l[index].astype(int) + 2*d_idx

            elif dataset_name.lower() == 'paip':
                label_path=os.path.join(dataset_root_organ,'label.csv')
                p, l = get_patient_label(label_path)
                index = [i for i in range(len(p))]
                random.shuffle(index)
                p = p[index]
                l = l[index].astype(int) + 2*d_idx

            elif dataset_name.lower() == 'tcga':
                label_path=os.path.join(dataset_root_organ,'label.csv')
                p, l = get_patient_label(label_path)
                index = [i for i in range(len(p))]
                random.shuffle(index)
                p = p[index]
                l = l[index].astype(int) + 2*d_idx


            if args.cv_fold > 1:
                train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

                train_p_list.append(train_p)
                train_l_list.append(train_l)
                test_p_list.append(test_p)
                test_l_list.append(test_l)
                val_p_list.append(val_p)
                val_l_list.append(val_l)


        # self.train_set = C16Dataset_Joint(self.train_p[k],self.train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True, patch_labels=True)
        # self.test_set = C16Dataset_Joint(self.test_p[k],self.test_l[k],root=args.dataset_root,persistence=args.persistence,\
        #                                     keep_same_psize=args.same_psize, patch_labels=True, return_coords=True)

        # if args.cv_fold > 1:
        #     train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

        one_fold(args,k,train_p_list, train_l_list, test_p_list, test_l_list,val_p_list,val_l_list,dataset_root_organ_list)



if __name__ == '__main__':
    main()
