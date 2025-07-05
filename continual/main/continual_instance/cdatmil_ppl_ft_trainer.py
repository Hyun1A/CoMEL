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


from main.continual_instance.abmil_ft_trainer import ABMIL_FT_Trainer

class CDATMIL_PPL_FT_Trainer(ABMIL_FT_Trainer):
    def __init__(self, args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list):
        super().__init__(args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list)

        self.T = 1.0
        self.tau1 = 0.95
        self.tau = 0.5
        self.tau2 = self.tau


    def set_dataset(self, args, k, task):
        if args.datasets.lower() == 'cm16,paip':

            dataset_root = self.dataset_root_organ_list[task]
            self.train_set = C16Dataset(self.train_p[task][k],self.train_l[task][k],root=dataset_root,persistence=args.persistence,\
                            keep_same_psize=args.same_psize,is_train=True, patch_labels=True, return_coords=True)
            
            test_sets, val_sets = [], []

            for d_idx, (test_p, test_l, val_p, val_l) in enumerate(zip(self.test_p,self.test_l,self.val_p,self.val_l)):
                dataset_root = self.dataset_root_organ_list[d_idx]
                test_set = C16Dataset(test_p[k],test_l[k],root=dataset_root,persistence=args.persistence,\
                                                    keep_same_psize=args.same_psize, patch_labels=True, return_coords=True)
                
                if args.val_ratio != 0.:
                    val_set = C16Dataset(val_p[k],val_l[k],root=dataset_root,persistence=args.persistence,\
                                                    keep_same_psize=args.same_psize, patch_labels=True, return_coords=True)
                else:
                    val_set = test_set

                test_sets.append(test_set)
                val_sets.append(val_set)

                if d_idx == task:
                    break
            
            self.test_set = test_sets
            self.val_set = val_sets


            self.joint_test_set = C16Dataset_Joint(test_sets)
            self.joint_val_set = C16Dataset_Joint(val_sets)


        return self.train_set, self.val_set, self.test_set,







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
        quality = 0.
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

                if args.model in ('cdatmil'):
                    train_logits, train_patch_attn, feat_bag, feat_inst = model(bag,return_attn=True,no_norm=False,return_inst=True)    
                    train_patch_attn = train_patch_attn.permute(1,0)
                    
                    logit_min, _ = train_patch_attn.min(dim=0, keepdim=True)
                    logit_max, _ = train_patch_attn.max(dim=0, keepdim=True)
                    train_patch_prob = (train_patch_attn-logit_min) / (logit_max+1e-10)
                    train_patch_prob = torch.cat([1-train_patch_prob, train_patch_prob],dim=1)
                    cls_loss,patch_num,keep_num = 0.,0.,0.       


                ############### Model forward #################
                ###############################################



                ###############################################
                ################ Compute loss #################
                if args.loss == 'ce':
                    logit_loss = criterion(train_logits.view(batch_size,-1),label)
                    val, pred = torch.softmax(train_logits.detach(), dim=1).max(dim=1)

                    n_slides += pred.size(0)

                    quality += val*(pred==label).float()

                    if i == 0:
                        train_acc = (label == pred).float().sum() / n_slides
                    else:
                        train_acc = ((n_slides - pred.size(0)) / n_slides) * train_acc + (1/n_slides) * (label == pred).float().sum() 


                    ##########################################
                    ################## PPL ###################
                    if ("paip" in self.args.datasets) or ("cm16" in self.args.datasets):
                        is_correct = ( pred == label )
                        is_tumor = label % 2 == 1
                        
                        try:
                            if not (is_tumor and is_correct): 
                                patch_logit_loss = torch.zeros_like(logit_loss).to(feat_bag.device)

                            else:
                                bs, t, d = feat_inst.size()
                                feat_inst = feat_inst.reshape(bs*t, d)
                                
                                train_patch_attn_rev = (1-train_patch_attn) / (1-train_patch_attn).sum(dim=0)
                                feat_bag_rev = train_patch_attn_rev.T @ feat_inst 
                                
                                feat_inst_norm = feat_inst / feat_inst.norm(dim=1, keepdim=True)
                                feat_bag_norm = feat_bag / feat_bag.norm(dim=1, keepdim=True)
                                feat_bag_rev_norm = feat_bag_rev / feat_bag_rev.norm(dim=1, keepdim=True)
                                
                                feat_bag_disc = torch.cat([feat_bag_rev_norm, feat_bag_norm], dim=0)


                                disc_logit = feat_inst_norm @ feat_bag_disc.T
                                disc_sel = torch.softmax(disc_logit/self.T, dim=1)
                                disc_val, disc_pred = disc_sel.max(dim=1)
                                
                                val_idx = (val>self.tau1)*(disc_val > self.tau2)
                                
                                sel_disc_pred = disc_pred[val_idx]
                                sel_train_patch_prob = train_patch_prob[val_idx]

                                patch_log_softmax= torch.log(sel_train_patch_prob+1e-10)

                                if patch_log_softmax.size(0) > 0:
                                    patch_logit_loss = ( patch_log_softmax.size(0)/(bs*t) ) * F.nll_loss(patch_log_softmax, sel_disc_pred)                    
                                else:
                                    patch_logit_loss = torch.zeros_like(logit_loss).to(feat_bag.device)
                        except:
                            breakpoint()

                    ################## PPL ###################
                    ##########################################

                elif args.loss == 'bce':
                    logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))
                    patch_logit_loss = criterion(train_patch_prob, one_hot(patch_labels, num_classes=2))

                train_loss = args.cls_alpha * logit_loss +  cls_loss*args.aux_alpha

                ########### Include instance loss after wamup iters ###########
                if epoch >= args.train_patch_epoch:
                    train_loss += args.seg_coef * patch_logit_loss

                train_loss = train_loss / args.accumulation_steps
                ################ Compute loss #################
                ###############################################


            if pred.size(0) > 1:
                if i % 25 == 0:
                    print(f"[{i}/{len(loader)}] label: {label}, logit loss: {args.cls_alpha * logit_loss:15.5}, patch logit loss: {args.seg_coef * patch_logit_loss:15.5}, pred: {pred}, conf: {val}, train_acc: {train_acc}")
            else:
                if i % 25 == 0:
                    print(f"[{i}/{len(loader)}] label: {label.item()}, logit loss: {args.cls_alpha * logit_loss:15.5}, patch logit loss: {args.seg_coef * patch_logit_loss:15.5}, pred: {pred.item()}, conf: {val.item():15.5}, train_acc: {train_acc:15.5}")
    

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

        self.tau2 = (1/2)*(1+self.tau * (quality / n_slides))
        print("quality:", quality, "n_slides:", n_slides, "tau2:", self.tau2)


        end = time.time()
        train_loss_log = train_loss_log/len(loader)
        if not args.lr_supi and scheduler is not None:
            scheduler.step()
        
        return train_loss_log,start,end
