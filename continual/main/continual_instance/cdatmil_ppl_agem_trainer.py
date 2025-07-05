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







from cl_utils.buffer import Buffer
import quadprog as solver
import qpsolvers as solver


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger



def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params:
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            try:
                grads[begin: end].copy_(param.grad.data.view(-1))
            except:
                breakpoint()
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params:
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = solver.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))








class CDATMIL_PPL_AGEM_Trainer(CDATMIL_PPL_FT_Trainer):
    def __init__(self, args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list):
        super().__init__(args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.grad_dims = []

        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                self.grad_dims.append(param.data.numel())



        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)




    def after_task(self):
        self.current_task += 1

        loader = self.train_loader
        data = next(iter(loader))[1:]
        
        cur_x = data[0]
        cur_y = data[1]

        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )



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

            bs, t, d = bag.size()
            
            if self.current_task > 0:                
                buf_bag, buf_label = self.buffer.get_data(bs)
                buf_bag = buf_bag[0].unsqueeze(0)

            patch_labels = data[2].squeeze(0).squeeze(-1).to(device)        
                    
            
            n_slides, quality, pred, val, train_loss, logit_loss, patch_logit_loss, cls_loss, keep_num, patch_num, train_acc = \
                    self.observe(n_slides, quality, bag, amp_autocast, args, model, criterion, batch_size, label, patch_labels, epoch, train_acc, i)


            ######################################################
            ##################### optimize #######################
            if args.clip_grad > 0.:
                dispatch_clip_grad(
                    model_parameters(model),
                    value=args.clip_grad, mode='norm')

            if (i+1) % args.accumulation_steps == 0:
                train_loss.backward()
            ##################### optimize #######################
            ######################################################

            if self.current_task > 0:

                params = []
                for name, param in self.model.named_parameters():
                    if "classifier" not in name:
                        params.append(param.data.view(-1))

                store_grad(params, self.grad_xy, self.grad_dims)

                _, _, buf_pred, buf_val, buf_train_loss, buf_logit_loss, buf_patch_logit_loss, buf_cls_loss, buf_keep_num, buf_patch_num, _ = \
                        self.observe(n_slides, quality, buf_bag, amp_autocast, args, model, criterion, batch_size, buf_label, patch_labels, epoch, train_acc, i, is_buffer=True)

                ######################################################
                ##################### optimize #######################
                if args.clip_grad > 0.:
                    dispatch_clip_grad(
                        model_parameters(model),
                        value=args.clip_grad, mode='norm')

                if (i+1) % args.accumulation_steps == 0:
                    buf_train_loss.backward()
                ##################### optimize #######################
                ######################################################


                params = []
                for name, param in self.model.named_parameters():
                    if "classifier" not in name:
                        params.append(param.data.view(-1))

                store_grad(params, self.grad_er, self.grad_dims)

                dot_prod = torch.dot(self.grad_xy, self.grad_er)

                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)

                    params = []
                    for name, param in self.model.named_parameters():
                        if "classifier" not in name:
                            params.append(param.data.view(-1))

                    overwrite_grad(params, g_tilde, self.grad_dims)
                else:

                    params = []
                    for name, param in self.model.named_parameters():
                        if "classifier" not in name:
                            params.append(param.data.view(-1))

                    overwrite_grad(params, self.grad_xy, self.grad_dims)


                train_loss += buf_train_loss
                logit_loss += buf_logit_loss
                cls_loss += cls_loss


            ######################################################
            ##################### optimize #######################

            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()

            ##################### optimize #######################
            ######################################################



            if pred.size(0) > 1:
                if i % 25 == 0:
                    print(f"[{i}/{len(loader)}] label: {label}, logit loss: {args.cls_alpha * logit_loss:15.5}, patch logit loss: {args.seg_coef * patch_logit_loss:15.5}, pred: {pred}, conf: {val}, train_acc: {train_acc}")
            else:
                if i % 25 == 0:
                    print(f"[{i}/{len(loader)}] label: {label.item()}, logit loss: {args.cls_alpha * logit_loss:15.5}, patch logit loss: {args.seg_coef * patch_logit_loss:15.5}, pred: {pred.item()}, conf: {val.item():15.5}, train_acc: {train_acc:15.5}")
    


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

            #############################################
            ################# ER Buffer #################
            self.buffer.add_data(examples=data[0], labels=data[1])
            ################# ER Buffer #################
            #############################################



        self.tau2 = (1/2)*(1+self.tau * (quality / n_slides))
        print("quality:", quality, "n_slides:", n_slides, "tau2:", self.tau2)


        end = time.time()
        train_loss_log = train_loss_log/len(loader)
        if not args.lr_supi and scheduler is not None:
            scheduler.step()
        
        return train_loss_log,start,end



    def observe(self, n_slides, quality, bag, amp_autocast, args, model, criterion, batch_size, label, patch_labels, epoch, train_acc, i, is_buffer=False):
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

            ########### Include instance loss after wamup ites ###########
            if epoch >= args.train_patch_epoch:
                train_loss += args.seg_coef * patch_logit_loss

            train_loss = train_loss / args.accumulation_steps
            ################ Compute loss #################
            ###############################################
    



        return n_slides, quality, pred, val, train_loss, logit_loss, patch_logit_loss, cls_loss, keep_num, patch_num, train_acc