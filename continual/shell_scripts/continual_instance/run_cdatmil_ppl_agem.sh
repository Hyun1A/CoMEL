SETUP=continual_instance

BACKBONE=uni
MAGNIFICATION=20x

DATASETS='cm16,paip'
ORGAN='mixed,HCC,prostate,HBP,colon'

MODEL=cdatmil
CL_METHOD=agem
RANK=8
NLAYER=2
USE_PPL=True



PROJECT_NAME=${SETUP}_${BACKBONE}_${MAGNIFICATION}_${DATASETS}_${ORGAN}
TITLE=${MODEL}_ppl_${CL_METHOD}

OUTPUT_PATH=./output
MODEL_PATH="${OUTPUT_PATH}/${SETUP}/${BACKBONE}_${MAGNIFICATION}/${DATASETS}_${ORGAN}/${MODEL}_${NLAYER}"
PRETRAIN_MODEL_PATH="${MODEL_PATH}}/ckp.pt"
ROOT_DIR="/home/ldlqudgus756/cl_pathology/data"

DATASET_PATH="${ROOT_DIR}/#dataset#/#organ#/extract_patch256_256_w_patch_labels_mixed/${BACKBONE}"
WSI_PATH="${ROOT_DIR}/#dataset#/#organ#/WSIs"

CUDA_VISIBLE_DEVICES=0 python3 ./main/continual_instance/run.py \
    --project=${PROJECT_NAME} \
    --datasets=${DATASETS} \
    --organ=${ORGAN} \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --dataset_root=${DATASET_PATH} \
    --model_path=${MODEL_PATH} \
    --pool=attn \
    --n_trans_layers=${NLAYER} \
    --da_act=tanh \
    --title=${TITLE} \
    --epeg_k=15 \
    --crmsa_k=1 \
    --all_shortcut \
    --seed=2021 \
    --lr 1e-4 \
    --cl_lr 1e-4 \
    --rank ${RANK} \
    --wo_pretrain \
    --train_patch_epoch=0 \
    --cls_alpha 1 \
    --loss "ce" \
    --wsi_path ${WSI_PATH} \
    --ssl_config_file ./ssl_engine/ssl_config/histo_mil/supervised/supervised_paip_mil.yaml \
    --per_seg_vis_epoch 4 \
    --disable_vis_seg \
    --vis_sample_interval 32 \
    --noise_scale 0.005 \
    --input_dim 1024 \
    --wandb \
    --cv_fold=5 \
    --fold_start 0 \
    --fold_end 1 \
    --n_tasks 5 \
    --st_task 0 \
    --n_classes 2 \
    --num_epoch 100 \
    --val_interval 50 \
    --task_setup "continual" \
    --model ${MODEL} \
    --cl_method ${CL_METHOD} \
    --use_ppl ${USE_PPL} \
    --seg_coef 1.0 \
    --buffer_size 30





