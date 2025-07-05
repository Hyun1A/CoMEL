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
from modules import attmil_ft, rrt_ft

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


from main.continual_default.abmil_ft_trainer import ABMIL_FT_Trainer as ABMIL_FT_Trainer_Default

class ABMIL_FT_Trainer(ABMIL_FT_Trainer_Default):
    def __init__(self, args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list):
        super().__init__(args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list)



    def train_loop(self, args,model,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,epoch, criterion_reg=None):
        start = time.time()
        loss_cls_meter = AverageMeter()
        loss_cl_meter = AverageMeter()
        patch_num_meter = AverageMeter()
        keep_num_meter = AverageMeter()

        train_loss_log = 0.
        model.train()

        train_acc = 0.
        n_slides = 0.
        for i, data in enumerate(loader):
            optimizer.zero_grad()

            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)
                
            label=data[1].to(device)
            patch_labels = data[2].squeeze(0).squeeze(-1).to(device)        
                    
            with amp_autocast():
                ###############################################
                ############## Bag augmentation ###############
                if args.patch_shuffle:
                    bag = patch_shuffle(bag,args.shuffle_group)
                elif args.group_shuffle:
                    bag = group_shuffle(bag,args.shuffle_group)
                ############## Bag augmentation ###############
                ###############################################

                ###############################################
                ############### Model forward #################

                if args.model in ('clam_sb','clam_mb','dsmil'):
                    train_logits,cls_loss,patch_num,train_patch_attn = model(bag,label,criterion)  
                    keep_num = patch_num

                elif args.model in ('attmil', 'transmil', 'rrtmil', 'cdatmil'):
                    train_logits, train_patch_attn = model(bag,return_attn=True,no_norm=False)
                    cls_loss,patch_num,keep_num = 0.,0.,0.       


                elif "AgentMIL" in model.__class__.__name__:
                    train_logits, train_patch_attn = model(bag, return_attn=True, train_mode=True)      
                    cls_loss,patch_num,keep_num = 0.,0.,0.           

                ############### Model forward #################
                ###############################################

                ###############################################
                ################ Compute loss #################
                if args.loss == 'ce':
                    logit_loss = criterion(train_logits.view(batch_size,-1),label)
                    val, pred = torch.softmax(train_logits.detach(), dim=1).max(dim=1)

                    n_slides += pred.size(0)
                    if i == 0:
                        train_acc = (label == pred).float().sum() / n_slides
                    else:
                        train_acc = ((n_slides - pred.size(0)) / n_slides) * train_acc + (1/n_slides) * (label == pred).float().sum() 

                elif args.loss == 'bce':
                    logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))

                train_loss = args.cls_alpha * logit_loss +  cls_loss*args.aux_alpha


                train_loss = train_loss / args.accumulation_steps
                ################ Compute loss #################
                ###############################################


            if pred.size(0) > 1:
                if i % 25 == 0:
                    print(f"[{i}/{len(loader)}] label: {label}, logit loss: {args.cls_alpha * logit_loss:15.5}, pred: {pred}, conf: {val}, train_acc: {train_acc}")
            else:
                if i % 25 == 0:
                    print(f"[{i}/{len(loader)}] label: {label.item()}, logit loss: {args.cls_alpha * logit_loss:15.5}, pred: {pred.item()}, conf: {val.item():15.5}, train_acc: {train_acc:15.5}")
    

            ######################################################
            ##################### optimize #######################
            if args.clip_grad > 0.:
                dispatch_clip_grad(
                    model_parameters(model),
                    value=args.clip_grad, mode='norm')

            if (i+1) % args.accumulation_steps == 0:
                train_loss.backward()


                optimizer.step()
                if args.lr_supi and scheduler is not None:
                    scheduler.step()
            ##################### optimize #######################
            ######################################################



            ###########################################
            ################# Logging #################
            loss_cls_meter.update(logit_loss,1)
            loss_cl_meter.update(cls_loss,1)
            patch_num_meter.update(patch_num,1)
            keep_num_meter.update(keep_num,1)

            if (i+1) % args.log_iter == 0 or i == len(loader)-1:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                rowd = OrderedDict([
                    ('cls_loss',loss_cls_meter.avg),
                    ('lr',lr),
                    ('cl_loss',loss_cl_meter.avg),
                    ('patch_num',patch_num_meter.avg),
                    ('keep_num',keep_num_meter.avg),
                    ('train_slide_acc',train_acc),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                if args.wandb:
                    wandb.log(rowd)

            train_loss_log = train_loss_log + train_loss.item()
            ################# Logging #################
            ###########################################


        end = time.time()
        train_loss_log = train_loss_log/len(loader)
        if not args.lr_supi and scheduler is not None:
            scheduler.step()
        
        return train_loss_log,start,end