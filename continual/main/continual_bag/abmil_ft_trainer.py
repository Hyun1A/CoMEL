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


        elif args.datasets.lower() == 'tcga':

            dataset_root = self.dataset_root_organ_list[task]
            self.train_set = TCGADataset(self.train_p[task][k],self.train_l[task][k],args.tcga_max_patch,dataset_root,persistence=args.persistence,\
                    keep_same_psize=args.same_psize,is_train=True, patch_labels=False, return_coords=True)
            
            test_sets, val_sets = [], []


            for d_idx, (test_p, test_l, val_p, val_l) in enumerate(zip(self.test_p,self.test_l,self.val_p,self.val_l)):
                dataset_root = self.dataset_root_organ_list[d_idx]
                test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,dataset_root,persistence=args.persistence,\
                        keep_same_psize=args.same_psize, patch_labels=False, return_coords=True)
                
                if args.val_ratio != 0.:
                    val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,dataset_root,persistence=args.persistence,\
                        keep_same_psize=args.same_psize, patch_labels=False, return_coords=True)
                else:
                    val_set = test_set        

                test_sets.append(test_set)
                val_sets.append(val_set)

                if d_idx == task:
                    break
            
            self.test_set = test_sets
            self.val_set = val_sets


            self.joint_test_set = TCGADataset_Joint(test_sets)
            self.joint_val_set = TCGADataset_Joint(val_sets)


        return self.train_set, self.val_set, self.test_set,




    def train(self):
        #########################################################
        ################## for results per taks #################

        results_per_task = dict()
        for task in range(self.current_task+1):
            task_name = self.task_list[task]
            results_per_task[task_name] = {"bag_result": [],
                                         "inst_result":[],
                                         "opt_result":[]}
        
        ################## for results per taks #################            
        #########################################################


        #########################################################
        ################# for results over tasks ################

        acs,pre,rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]

        ################# for results over tasks ################
        #########################################################

        epoch_start = 0

        if self.args.fix_train_random:
            seed_torch(self.args.seed)

        train_time_meter = AverageMeter()
        for epoch in range(epoch_start, self.args.num_epoch):
            train_loss,start,end = self.train_loop(self.args,self.model,self.train_loader,\
                                            self.optimizer,self.device,self.amp_autocast,\
                                            self.criterion,self.loss_scaler,self.scheduler,\
                                            self.k, epoch, self.criterion_reg)
                    
            train_time_meter.update(end-start)

            default_wsi_path = deepcopy(self.args.wsi_path)

            self.args.wsi_path = default_wsi_path.replace("#organ#", self.organ)


            if not ((epoch+1)%self.args.val_interval==0 or epoch == self.args.num_epoch-1):
                if not self.args.no_log:
                    print(f'Epoch [{epoch+1}/{self.args.num_epoch}]')


            else:
                accuracy_t, auc_value_t, precision_t, recall_t, fscore_t = [],[],[],[],[]

                for task in range(self.current_task+1):
                    task_name = self.task_list[task]

                    val_loader = self.val_loader[task]


                    stop, accuracy, auc_value, precision, recall, fscore, test_loss, vis_data = \
                                    self.val_loop(self.args,self.model,val_loader,self.device,self.criterion,self.early_stopping,epoch)


                    #####################################
                    ##### separate results per task #####
                    results_per_task[task_name]["bag_result"].append([accuracy, auc_value, precision, recall, fscore])
                    ##### separate results per task #####
                    #####################################


                    ###########################################################################
                    #################### log of results per task to wandb #####################
                    if vis_data is not None:
                        img_log_dict = {}

                        for img, wsi_name in zip(vis_data[-2], vis_data[-1]):
                            img_log_dict[str(self.k)+f'-fold_image_{task}_task_{task_name}/'+wsi_name] = wandb.Image(img)

                        wandb.log(img_log_dict)

                    # breakpoint()

                    if self.args.wandb:
                        rowd = OrderedDict([
                            ("val_acc",accuracy),
                            ("val_precision",precision),
                            ("val_recall",recall),
                            ("val_fscore",fscore),
                            ("val_auc",auc_value),
                            ("val_loss",test_loss),
                            ("epoch",epoch),
                        ])

                        rowd = OrderedDict([ (str(self.k)+f'-fold_{task}_task_{task_name}/'+_k,_v) for _k, _v in rowd.items()])
                        wandb.log(rowd,commit=False)
                    #################### log of results per task to wandb #####################
                    ###########################################################################




                    #####################################
                    ######## result over tasks  #########
                    accuracy_t.append(accuracy)
                    auc_value_t.append(auc_value)
                    precision_t.append(precision)
                    recall_t.append(recall)
                    fscore_t.append(fscore)


                    ######## result over tasks  #########
                    #####################################


                #####################################
                ######## result over tasks  #########
                accuracy = np.array(accuracy_t).mean().item()
                auc_value = np.array(auc_value_t).mean().item()
                precision = np.array(precision_t).mean().item()
                recall = np.array(recall_t).mean().item()
                fscore = np.array(fscore_t).mean().item()

                # test_loss = np.array(test_loss_t).mean().item()
                # vis_data = np.array(vis_data_t).mean().item()

                acs.append(accuracy)
                pre.append(precision)
                rec.append(recall)
                fs.append(fscore)
                auc.append(auc_value)     
                ######## result over tasks  #########
                #####################################


                if not self.args.no_log:
                    print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f, time: %.3f(%.3f)' % 
                (epoch+1, self.args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))



                ###########################################################################
                ################### log of results over tasks to wandb ####################
                if self.args.wandb:
                    rowd = OrderedDict([
                        ("val_acc",accuracy),
                        ("val_precision",precision),
                        ("val_recall",recall),
                        ("val_fscore",fscore),
                        ("val_auc",auc_value),
                        ("val_loss",test_loss),
                        ("epoch",epoch),
                    ])

                    rowd = OrderedDict([ (str(self.k)+f'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd,commit=False)
                ################### log of results over tasks to wandb ####################
                ###########################################################################


        ckc_metric = [acs,auc,pre,rec,fs,te_auc,te_fs]
        results = {"ckc_metric": ckc_metric,"k": self.k}
        results_per_task["k"] = self.k
        ckp = {'model': self.model.state_dict(),'k': self.k}

        if self.args.task_setup in ["single", "joint"]:
            os.makedirs(self.args.model_path, exist_ok=True)
            torch.save(ckp, os.path.join(self.args.model_path, f'fold_{self.k}_final_model.pt'))
            torch.save(results, os.path.join(self.args.model_path, f'fold_{self.k}_final_ckc_metric.pt'))

        elif self.args.task_setup in ["continual"]:
            os.makedirs(self.args.model_path, exist_ok=True)
            torch.save(ckp, os.path.join(self.args.model_path, f'fold_{self.k}_task_{self.current_task}_final_model.pt'))
            torch.save(results, os.path.join(self.args.model_path, f'fold_{self.k}_task_{self.current_task}_final_ckc_metric.pt'))     
            torch.save(results_per_task, os.path.join(self.args.model_path, f'fold_{self.k}_task_{self.current_task}_final_ckc_metric_per_task.pt'))     







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
    



    def val_loop(self, args,model,loader,device,criterion,early_stopping,epoch,labels=[0,1]):
        model.eval()
        loss_cls_meter = AverageMeter()
        bag_logit, bag_labels=[], []
        
        bag_coords, bag_wsi_names = [], []
        
        # pred= []
        with torch.no_grad():
            for i, data in enumerate(loader):
                if len(data[1]) > 1:
                    bag_labels.extend(data[1])
                    bag_coords += [data[2][i] for i in range(len(data[3]))]
                    bag_wsi_names += [data[3][i] for i in range(len(data[4]))]
                    
                else:
                    bag_labels.append(data[1])
                    bag_coords.append(data[2])
                    bag_wsi_names.append(data[3])                

                if isinstance(data[0],(list,tuple)):
                    for i in range(len(data[0])):
                        data[0][i] = data[0][i].to(device)
                    bag=data[0]
                    batch_size=data[0][0].size(0)
                else:
                    bag=data[0].to(device)  # b*n*1024
                    batch_size=bag.size(0)

                label=data[1].to(device)
                

                if args.model == 'dsmil':
                    test_logits,cls_loss,patch_num,_ = model(bag,label,criterion)
                    keep_dim=patch_num


                elif args.model in ["attmil", "transmil", "rrtmil", "cdatmil"]:
                    test_logits, _ = model(bag,return_attn=True,no_norm=False)
                    cls_loss,patch_num,keep_num = 0.,0.,0.       


                elif "AgentMIL" in model.__class__.__name__:
                    test_logits, _ = model(bag, return_attn=True, train_mode=False)      
                    cls_loss,patch_num,keep_num = 0.,0.,0.       

                else:
                    test_logits, test_patch_logits = model(bag, return_patch_logits=True)

                if args.loss == 'ce':
                    if args.model in ["attmil", "dsmil", "transmil", "rrtmil", "agentmil", "cdatmil"]:
                        test_loss = criterion(test_logits.view(batch_size,-1),label)
                        if batch_size > 1:
                            bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze())
                        else:
                            bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze())

                    
                    else:
                        test_loss = criterion(test_logits.view(batch_size,-1),label)
                        if batch_size > 1:
                            bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze())
                        else:
                            bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze())
                        
                loss_cls_meter.update(test_loss,1)
        
        subtyping = (args.datasets.lower() == "tcga")  

        if self.args.task_setup in ["single", "joint"]:
            labels = list(range(self.args.n_classes))
        elif self.args.task_setup in ["continual"]:
            labels = self.labels

        accuracy, auc_value, precision, recall, fscore = \
                                    five_scores(bag_labels, bag_logit, subtyping, return_pred_patch=True, labels=labels)    
        
        vis_data = None
        if (not self.args.disable_vis_seg) and (epoch % args.per_seg_vis_epoch == 0):
            vis_data = visualize_prediction(bag_labels_patch, bag_predictions_patch, bag_coords,\
                        bag_wsi_names,wsi_path=args.wsi_path, vis_sample_interval=args.vis_sample_interval)

        # early stop
        if early_stopping is not None:
            early_stopping(epoch,-auc_value,model)
            stop = early_stopping.early_stop
        else:
            stop = False
        
        return stop, accuracy, auc_value, precision, recall, fscore, loss_cls_meter.avg, vis_data
