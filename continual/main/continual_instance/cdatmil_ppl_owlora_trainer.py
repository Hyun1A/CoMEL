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


from main.continual_instance.cdatmil_ppl_ft_trainer import CDATMIL_PPL_FT_Trainer

class CDATMIL_PPL_OWLoRA_Trainer(CDATMIL_PPL_FT_Trainer):
    def __init__(self, args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list):
        super().__init__(args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list)


        ##################################
        ############ for CL ##############
        self.checkpoint = None
        self.fish = None

        self.lambda3 = 1.0

        ############ for CL ##############
        ##################################




    def set_optimizer(self, args):
        # optimizer

        if self.current_task == 0:
            if args.opt == 'adamw':
                self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            elif args.opt == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

            if args.lr_sche == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.num_epoch, 0) \
                                if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.num_epoch*len(self.train_loader), 0)
            elif args.lr_sche == 'cosine_restart':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, args.num_epoch//args.num_restart_cycles, 1, 0)
            elif args.lr_sche == 'step':
                assert not args.lr_supi
                # follow the DTFD-MIL
                # ref:https://github.com/hrzhang1123/DTFD-MIL
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,args.num_epoch / 2, 0.2)
            elif args.lr_sche == 'const':
                self.scheduler = None

            if args.early_stopping:
                self.early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, \
                                                stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,
                                                save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
            else:
                self.early_stopping = None


        else:
            # self.train_params = dict()

            train_params = []
            for name, module in self.model.update_modules.items():
                train_params += list(module.lora_layers[-1].parameters())

            for name, params in self.model.named_parameters():
                if "classifier" in name:
                    train_params += [params]

            
            if args.opt == 'adamw':
                self.optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
            elif args.opt == 'adam':
                self.optimizer = torch.optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)

            if args.lr_sche == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.num_epoch, 0) \
                                if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.num_epoch*len(self.train_loader), 0)
            elif args.lr_sche == 'cosine_restart':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, args.num_epoch//args.num_restart_cycles, 1, 0)
            elif args.lr_sche == 'step':
                assert not args.lr_supi
                # follow the DTFD-MIL
                # ref:https://github.com/hrzhang1123/DTFD-MIL
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,args.num_epoch / 2, 0.2)
            elif args.lr_sche == 'const':
                self.scheduler = None

            if args.early_stopping:
                self.early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, \
                                                stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,
                                                save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
            else:
                self.early_stopping = None





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


                    ################## PPL ###################
                    ##########################################

                elif args.loss == 'bce':
                    logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))
                    patch_logit_loss = criterion(train_patch_prob, one_hot(patch_labels, num_classes=2))

                train_loss = args.cls_alpha * logit_loss +  cls_loss*args.aux_alpha

                ########### Include instance loss after wamup iters ###########
                if epoch >= args.train_patch_epoch:
                    train_loss += args.seg_coef * patch_logit_loss

                ########### CL loss ############



                penalty = self.penalty()
                train_loss += self.lambda3 * penalty

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


                if self.current_task > 0:
                    for name, module in self.model.update_modules.items():

                        down_weight = module.lora_layers[-1].down.weight
                        down_weight_ortho_grad = None
                        for layer in module.lora_layers[:-1]:
                            down_ortho_mat = layer.down.weight
                            down_ortho_space = down_ortho_mat.T @ down_ortho_mat

                            if down_weight_ortho_grad == None:
                                down_weight_ortho_grad = down_weight.grad @ down_ortho_space
                            else:
                                down_weight_ortho_grad += down_weight.grad @ down_ortho_space

                        down_weight.grad = down_weight.grad - down_weight_ortho_grad


                        up_weight = module.lora_layers[-1].up.weight
                        up_weight_ortho_grad = None
                        for layer in module.lora_layers[:-1]:
                            up_ortho_mat = layer.up.weight
                            up_ortho_space = up_ortho_mat @ up_ortho_mat.T

                            if up_weight_ortho_grad == None:
                                up_weight_ortho_grad =  up_ortho_space @ up_weight.grad
                            else:
                                up_weight_ortho_grad += up_ortho_space @ up_weight.grad

                        up_weight.grad = up_weight.grad - up_weight_ortho_grad


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







    def penalty(self):
        if self.current_task == 0:
            return torch.tensor(0.0).to(self.device)
        else:
            intra_ortho_loss = 0.
            for name, module in self.model.update_modules.items():
                lora_layer = module.lora_layers[-1]
                down_weight = lora_layer.down.weight
                up_weight = lora_layer.up.weight

                rank = down_weight.size(0)
                
                down_ortho = down_weight @ down_weight.T
                up_ortho = up_weight.T @ up_weight 
                identity_mat = torch.eye(rank).to(down_weight.device)

                ortho = ( ((down_ortho - identity_mat)**2).sum() + ((up_ortho - identity_mat)**2).sum() ) / rank**2
                intra_ortho_loss += ortho

            return intra_ortho_loss




    def after_task(self):
        if self.current_task ==0:
            self.model.save_initial_basis()

        self.model.expand_lora()
        self.current_task += 1

