import argparse


def get_config():
    # parser = argparse.ArgumentParser(description='MIL Training Script')
    parser = argparse.ArgumentParser(description="Configure MIL, SSL, and CL settings")

    ##############################################################################################
    ######################################### MIL configs ########################################
    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    parser.add_argument('--dataset_root', default='/data/xxx/TransMIL', type=str, help='Dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')
    
    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--aux_alpha', default=1.0, type=float, help='Auxiliary loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce, proxy_anchor]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='rrtmil', type=str, help='Model name')
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--only_rrt_enc',action='store_true', help='RRT+other MIL models [dsmil,clam,]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    # Transformer
    parser.add_argument('--attn', default='rmsa', type=str, help='Inner attention')
    parser.add_argument('--pool', default='attn', type=str, help='Classification poolinp. use abmil.')
    parser.add_argument('--ffn', action='store_true', help='Feed-forward network. only for ablation')
    parser.add_argument('--n_trans_layers', default=2, type=int, help='Number of layer in the transformer')
    parser.add_argument('--mlp_ratio', default=4., type=int, help='Ratio of MLP in the FFN')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--all_shortcut', action='store_true', help='x = x + rrt(x)')
    # R-MSA
    parser.add_argument('--region_attn', default='native', type=str, help='only for ablation')
    parser.add_argument('--min_region_num', default=0, type=int, help='only for ablation')
    parser.add_argument('--region_num', default=8, type=int, help='Number of the region. [8,12,16,...]')
    parser.add_argument('--trans_dim', default=64, type=int, help='only for ablation')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the R-MSA')
    parser.add_argument('--trans_drop_out', default=0.1, type=float, help='Dropout in the R-MSA')
    parser.add_argument('--drop_path', default=0., type=float, help='Droppath in the R-MSA')
    # PEG or PPEG. only for alation
    parser.add_argument('--pos', default='none', type=str, help='Position embedding, enable PEG or PPEG')
    parser.add_argument('--pos_pos', default=0, type=int, help='Position of pos embed [-1,0]')
    parser.add_argument('--peg_k', default=7, type=int, help='K of the PEG and PPEG')
    parser.add_argument('--peg_1d', action='store_true', help='1-D PEG and PPEG')
    # EPEG
    parser.add_argument('--epeg', action='store_false', help='enable epeg')
    parser.add_argument('--epeg_bias', action='store_false', help='enable conv bias')
    parser.add_argument('--epeg_2d', action='store_true', help='enable 2d conv. only for ablation')
    parser.add_argument('--epeg_k', default=15, type=int, help='K of the EPEG. [9,15,21,...]')
    parser.add_argument('--epeg_type', default='attn', type=str, help='only for ablation')
    # CR-MSA
    parser.add_argument('--cr_msa', action='store_false', help='enable CR-MSA')
    parser.add_argument('--crmsa_k', default=3, type=int, help='K of the CR-MSA. [1,3,5]')
    parser.add_argument('--crmsa_heads', default=8, type=int, help='head of CR-MSA. [1,8,...]')
    parser.add_argument('--crmsa_mlp', action='store_true', help='mlp phi of CR-MSA?')

    # DAttention
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')
    
    # Custom 
    parser.add_argument('--pretrain_model_path', type=str, help='Pretrain model path')
    parser.add_argument('--wo_pretrain', action='store_true', help='choose whether to use pretrained model or not')
    parser.add_argument('--rank', type=int, help='rank for PEFT')
    parser.add_argument('--train_patch_epoch', type=int, default=3, help='rank for PEFT')
    parser.add_argument('--seg_coef', type=float, default=1., help='none')    
    parser.add_argument('--proxy_reg', type=float, default=0.1, help='none')    
    parser.add_argument('--mrg', type=float, default=0.1, help='none')    
    parser.add_argument('--temperature', type=float, default=32, help='none')    
    parser.add_argument('--pl_path', default="./", type=str, help='Output path')
    parser.add_argument('--noise_scale', type=float, default=0.05, help='none')    
    parser.add_argument('--wsi_path', type=str, default="", help='none') 
    parser.add_argument('--disable_vis_seg',action='store_true')    
    parser.add_argument('--per_seg_vis_epoch', type=int, default=10)    
    parser.add_argument('--vis_sample_interval', type=int, default=10)
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--fold_end', type=int, default=5)
    parser.add_argument('--prune_ratio', type=float, default=0.25)
    parser.add_argument('--num_restart_cycles', type=int, default=1)
    parser.add_argument('--organ', type=str, default="prostate", help='none') 
    parser.add_argument('--agent_num', type=int, default=128, help='none') 
    parser.add_argument('--task_setup', type=str, default="joint", help='none') 

    parser.add_argument('--val_interval', type=int, default=10, help='none') 


    ######################################### MIL configs ########################################
    ##############################################################################################





    ##############################################################################################
    ######################################### CL configs #########################################

    parser.add_argument('--cl_method', type=str, default="ewc", help='none')
    parser.add_argument('--n_tasks', type=int, default=3, help='none')
    parser.add_argument('--st_task', type=int, default=0, help='none') 
    parser.add_argument('--organ_lists', type=str, default="prostate,HCC,HBP", help='none') 
    parser.add_argument('--cl_lr', default=2e-4, type=float, help='Continual learning rate [0.0002]')
    
    parser.add_argument('--e_lambda', default=1.0, type=float, help='Continual learning rate [0.0002]')
    parser.add_argument('--e_gamma', default=0.9, type=float, help='Continual learning rate [0.0002]')

    parser.add_argument('--buffer_size', default=30, type=float, help='Continual learning rate [0.0002]')
    parser.add_argument('--der_alpha', default=3.0, type=float, help='Continual learning rate [0.0002]')
    parser.add_argument('--lwf_alpha', default=1.0, type=float, help='Continual learning rate [0.0002]')


    ######################################### CL configs #########################################
    ##############################################################################################





    ##############################################################################################
    ######################################### SSL configs ########################################
    
    """
    Saving & loading of the model.
    """
    parser.add_argument("-sn", "--save_name", type=str, default="fixmatch")
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)

    """
    Training Configuration of FixMatch
    """
    parser.add_argument(
        "--ema_m", type=float, default=0.999, help="ema momentum for eval_model"
    )
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)

    """
    Algorithms Configurations
    """

    ## core algorithm setting
    parser.add_argument(
        "-alg", "--algorithm", type=str, default="fixmatch", help="ssl algorithm"
    )

    ## imbalance algorithm setting
    parser.add_argument(
        "-imb_alg",
        "--imb_algorithm",
        type=str,
        default=None,
        help="imbalance ssl algorithm",
    )

    """
    Data Configurations
    """


    ## imbalanced setting arguments
    parser.add_argument(
        "--lb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of labeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of unlabeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_num_labels",
        type=int,
        default=None,
        help="number of labels for unlabeled data, used for determining the maximum "
        "number of labels in imbalanced setting",
    )

    # config file
    parser.add_argument("--ssl_config_file", type=str, default="")    
    
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="imbalance ratio of unlabeled data, default to 1",
    )

    parser.add_argument('--thresh_warmup', type=bool, default=False, help='threshold warmup for PPL')
    parser.add_argument('--p_cutoff', default=0.99, type=float, help='None')
    parser.add_argument('--hard_label', default=True, type=bool, help='None')
    parser.add_argument('--T', default=1.9, type=float, help='None')
    parser.add_argument('--ema_ca', default=0.99, type=float, help='None')
    parser.add_argument('--use_ppl', default=False, type=bool, help='None')
    
    ######################################### SSL configs ########################################
    ##############################################################################################




    # add algorithm specific parameters
    args = parser.parse_args()


    return args
    
    
