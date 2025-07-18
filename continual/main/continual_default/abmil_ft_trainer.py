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
from modules import attmil_ft, rrt_ft, cdatmil_ft, cdatmil_lora, cdatmil_inflora, cdatmil_owlora

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

class ABMIL_FT_Trainer(ABMIL_Trainer_Single):
    def __init__(self, args, k, train_p, train_l, test_p, test_l,val_p,val_l,dataset_root_organ_list):
        
        self.args = args
        self.organ = args.organ
        self.k =k
        self.train_p = train_p
        self.train_l = train_l
        self.test_p = test_p
        self.test_l = test_l
        self.val_p = val_p
        self.val_l = val_l
        

        self.task_list = self.organ.split(",")
        self.dataset_root_organ_list = dataset_root_organ_list
        self.n_tasks = self.args.n_tasks
        self.current_task = 0
        self.ths_patch_prev = []
        
        self.loss_scaler = GradScaler() if args.amp else None
        self.amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.set_model(args)
        
        if args.loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
            self.criterion_reg=None
        elif args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            self.criterion_reg=None



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




    def set_data_loader(self, args):
        # ---> Loading data
        if args.fix_loader_random:
            # generated by int(torch.empty((), dtype=torch.int64).random_().item())
            big_seed_list = 7784414403328510413
            generator = torch.Generator()
            generator.manual_seed(big_seed_list)  
            self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, sampler=RandomSampler(self.train_set), num_workers=args.num_workers)

        self.val_loader = []
        self.test_loader = []

        for task in range(self.current_task+1):
            self.val_loader.append( DataLoader(self.val_set[task], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) )
            self.test_loader.append( DataLoader(self.test_set[task], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) )
    

        self.joint_val_loader = DataLoader(self.joint_val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.joint_test_loader = DataLoader(self.joint_test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


        return self.train_loader, self.val_loader, self.test_loader






    def set_model(self, args):
        # bulid networks

        rrt_enc = None

        if args.model == 'rrtmil':
            model_params = {
                'input_dim': args.input_dim,
                'n_classes': args.n_classes,
                'dropout': args.dropout,
                'act': args.act,
                'region_num': args.region_num,
                'pos': args.pos,
                'pos_pos': args.pos_pos,
                'pool': args.pool,
                'peg_k': args.peg_k,
                'drop_path': args.drop_path,
                'n_layers': args.n_trans_layers,
                'n_heads': args.n_heads,
                'attn': args.attn,
                'da_act': args.da_act,
                'trans_dropout': args.trans_drop_out,
                'ffn': args.ffn,
                'mlp_ratio': args.mlp_ratio,
                'trans_dim': args.trans_dim,
                'epeg': args.epeg,
                'min_region_num': args.min_region_num,
                'qkv_bias': args.qkv_bias,
                'epeg_k': args.epeg_k,
                'epeg_2d': args.epeg_2d,
                'epeg_bias': args.epeg_bias,
                'epeg_type': args.epeg_type,
                'region_attn': args.region_attn,
                'peg_1d': args.peg_1d,
                'cr_msa': args.cr_msa,
                'crmsa_k': args.crmsa_k,
                'all_shortcut': args.all_shortcut,
                'crmsa_mlp':args.crmsa_mlp,
                'crmsa_heads':args.crmsa_heads,
            }
            model = rrt_ft.RRTMIL(**model_params).to(self.device)




        elif args.model == 'cdatmil':
            model_params = {
                'input_dim': args.input_dim,
                'n_classes': args.n_classes,
                'dropout': args.dropout,
                'act': args.act,
                'region_num': args.region_num,
                'pos': args.pos,
                'pos_pos': args.pos_pos,
                'pool': args.pool,
                'peg_k': args.peg_k,
                'drop_path': args.drop_path,
                'n_layers': args.n_trans_layers,
                'n_heads': args.n_heads,
                'attn': args.attn,
                'da_act': args.da_act,
                'trans_dropout': args.trans_drop_out,
                'ffn': args.ffn,
                'mlp_ratio': args.mlp_ratio,
                'trans_dim': args.trans_dim,
                'epeg': args.epeg,
                'min_region_num': args.min_region_num,
                'qkv_bias': args.qkv_bias,
                'epeg_k': args.epeg_k,
                'epeg_2d': args.epeg_2d,
                'epeg_bias': args.epeg_bias,
                'epeg_type': args.epeg_type,
                'region_attn': args.region_attn,
                'peg_1d': args.peg_1d,
                'cr_msa': args.cr_msa,
                'crmsa_k': args.crmsa_k,
                'all_shortcut': args.all_shortcut,
                'crmsa_mlp':args.crmsa_mlp,
                'crmsa_heads':args.crmsa_heads,
            }

            if self.args.cl_method == "lora":
                model = cdatmil_lora.CDATMIL(**model_params).to(self.device)
            
            elif self.args.cl_method == "inflora":
                model = cdatmil_inflora.CDATMIL(**model_params).to(self.device)
            
            elif self.args.cl_method == "owlora":
                model = cdatmil_owlora.CDATMIL(**model_params).to(self.device)
            else:
                model = cdatmil_ft.CDATMIL(**model_params).to(self.device)




        elif args.model == 'attmil':
            model = attmil_ft.DAttention(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act,rrt=rrt_enc).to(self.device)
        elif args.model == 'gattmil':
            model = attmil.AttentionGated(input_dim=args.input_dim,dropout=args.dropout,rrt=rrt_enc).to(self.device)
        elif args.model == 'ibmil':
            if not args.confounder_path.endswith('.npy'):
                _confounder_path = os.path.join(args.confounder_path,str(self.k),'train_bag_cls_agnostic_feats_proto_'+str(args.confounder_k)+'.npy')
            else:
                _confounder_path =args.confounder_path
            model = attmil_ibmil.Dattention_ori(out_dim=args.n_classes,dropout=args.dropout,in_size=args.input_dim,confounder_path=_confounder_path,rrt=rrt_enc).to(self.device)
        # follow the official code
        # ref: https://github.com/mahmoodlab/CLAM
        elif args.model == 'clam_sb':
            model = clam.CLAM_SB(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act,rrt=rrt_enc).to(self.device)
        elif args.model == 'clam_mb':
            model = clam.CLAM_MB(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act,rrt=rrt_enc).to(self.device)
        elif args.model == 'transmil':
            model = transmil.TransMIL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(self.device)
        elif args.model == 'dsmil':
            model = dsmil.MILNet(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act,rrt=rrt_enc).to(self.device)
            args.cls_alpha = 0.5
            args.aux_alpha = 0.5
            state_dict_weights = torch.load('./modules/mil_modules/init_ckp/dsmil_init.pth')
            info = model.load_state_dict(state_dict_weights, strict=False)
            if not args.no_log:
                print(info)
        elif args.model == 'meanmil':
            model = mean_max.MeanMIL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act,rrt=rrt_enc).to(self.device)
        elif args.model == 'maxmil':
            model = mean_max.MaxMIL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act,rrt=rrt_enc).to(self.device)

        self.model = model

        ############################################
        ########## load pre-trained model ##########
        
        if not args.wo_pretrain:
            ckp_pretrain = torch.load(args.pretrain_model_path)
            self.model.load_state_dict(ckp_pretrain['model'])
        ########## load pre-trained model ##########
        ############################################


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
        mious,dices,acs_patch,fs_patch,auc_patch,ths_patch=[],[],[],[],[],[]

        optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = 0,0,0,0,0,0
        opt_miou, opt_dice, opt_acc_patch, opt_auc_patch, opt_fs_patch = 0,0,0,0,0

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
                count_tumor_slide_t, miou_t, dice_t, acc_patch_t, auc1_patch_t, fscore_patch_t, th_patch_t, test_loss_t = [],[],[],[],[],[],[],[]

                for task in range(self.current_task+1):
                    task_name = self.task_list[task]

                    val_loader = self.val_loader[task]

                    th_patch_prev = self.ths_patch_prev[task] if task < self.current_task else None

                    stop, accuracy, auc_value, precision, recall, fscore, count_tumor_slide, miou, dice, acc_patch, auc1_patch, fscore_patch, th_patch, test_loss, vis_data = \
                                    self.val_loop(self.args,self.model,val_loader,self.device,self.criterion,self.early_stopping,epoch,th_patch=th_patch_prev)

                    if task == self.current_task:
                        self.ths_patch_prev.append(th_patch)


                    #####################################
                    ##### separate results per task #####
                    results_per_task[task_name]["bag_result"].append([accuracy, auc_value, precision, recall, fscore])
                    results_per_task[task_name]["inst_result"].append([count_tumor_slide, miou, dice, acc_patch, auc1_patch, fscore_patch, th_patch])
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
                            ("val_miou",miou),
                            ("val_dice",dice),
                            ("val_acc_patch",acc_patch),
                            ("val_ths_patch", th_patch),
                            ("val_fs_patch", fscore_patch),
                            ("val_auc_patch", auc1_patch),
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

                    count_tumor_slide_t.append(count_tumor_slide)
                    miou_t.append(miou)
                    dice_t.append(dice)
                    acc_patch_t.append(acc_patch)
                    auc1_patch_t.append(auc1_patch)
                    fscore_patch_t.append(fscore_patch)
                    th_patch_t.append(th_patch)
                    test_loss_t.append(test_loss)
                    # vis_data_t.append(vis_data)
                    ######## result over tasks  #########
                    #####################################


                #####################################
                ######## result over tasks  #########
                accuracy = np.array(accuracy_t).mean().item()
                auc_value = np.array(auc_value_t).mean().item()
                precision = np.array(precision_t).mean().item()
                recall = np.array(recall_t).mean().item()
                fscore = np.array(fscore_t).mean().item()
                count_tumor_slide = np.array(count_tumor_slide_t).mean().item()

                miou = np.array(miou_t).mean().item()
                dice = np.array(dice_t).mean().item()
                acc_patch = np.array(acc_patch_t).mean().item()
                auc1_patch = np.array(auc1_patch_t).mean().item()
                fscore_patch = np.array(fscore_patch_t).mean().item()
                th_patch = np.array(th_patch_t).mean().item()
                # test_loss = np.array(test_loss_t).mean().item()
                # vis_data = np.array(vis_data_t).mean().item()

                acs.append(accuracy)
                pre.append(precision)
                rec.append(recall)
                fs.append(fscore)
                auc.append(auc_value)
                mious.append(miou)
                dices.append(dice)
                acs_patch.append(acc_patch)
                ths_patch.append(th_patch)
                fs_patch.append(fscore_patch)
                auc_patch.append(auc1_patch)       
                ######## result over tasks  #########
                #####################################


                if not self.args.no_log:
                    print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f, miou: %.3f, dice: %.3f, time: %.3f(%.3f)' % 
                (epoch+1, self.args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, miou, dice, train_time_meter.val,train_time_meter.avg))



                ###########################################################################
                ################### log of results over tasks to wandb ####################
                if self.args.wandb:
                    rowd = OrderedDict([
                        ("val_acc",accuracy),
                        ("val_precision",precision),
                        ("val_recall",recall),
                        ("val_fscore",fscore),
                        ("val_auc",auc_value),
                        ("val_miou",miou),
                        ("val_dice",dice),
                        ("val_acc_patch",acc_patch),
                        ("val_ths_patch", th_patch),
                        ("val_fs_patch", fscore_patch),
                        ("val_auc_patch", auc1_patch),
                        ("val_loss",test_loss),
                        ("epoch",epoch),
                    ])

                    rowd = OrderedDict([ (str(self.k)+f'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd,commit=False)
                ################### log of results over tasks to wandb ####################
                ###########################################################################

        ckc_metric = [acs,auc,pre,rec,fs,mious,dices,acc_patch,auc_patch,fs_patch,ths_patch,te_auc,te_fs]
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



    def cl_train(self):
        for task in range(self.args.n_tasks):
            self.before_task()
            self.train()
            self.after_task()


    def before_task(self):
        self.labels = [2*self.current_task, 2*self.current_task+1]
        self.set_dataset(self.args, self.k, self.current_task)
        self.set_data_loader(self.args)
        self.update_model()
        self.set_optimizer(self.args)


    def after_task(self):
        self.current_task += 1


    def update_model(self):
        if self.current_task > 0 and self.args.model in ["attmil", "rrtmil", "cdatmil"]:
            cls_weight = self.model.classifier[-1].weight.data.clone()
            # cls_bias = self.model.classifier[-1].bias.data.clone()
            c,d=cls_weight.size()

            self.model.classifier.append( nn.Linear(in_features=d, out_features=len(self.labels)) )
            self.model.classifier[-1].to(cls_weight.device)
            # self.model.classifier[-1].weight.data = cls_weight
            # self.model.classifier[-1].bias.data = cls_bias

            self.model.classifier[-2].weight.requires_grad = False
            self.model.classifier[-2].bias.requires_grad = False
