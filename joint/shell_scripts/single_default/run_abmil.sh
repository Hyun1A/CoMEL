SETUP=single_default

BACKBONE=uni
MAGNIFICATION=20x

DATASETS=paip
ORGAN=prostate

MODEL=attmil
RANK=8
NLAYER=2

PROJECT_NAME=${SETUP}_${BACKBONE}_${MAGNIFICATION}_${DATASETS}_${ORGAN}_debug
TITLE=${MODEL}_seg_coef_10.0

OUTPUT_PATH=./output
MODEL_PATH="${OUTPUT_PATH}/${SETUP}/${BACKBONE}_${MAGNIFICATION}/${DATASETS}_${ORGAN}/${MODEL}_${NLAYER}"
PRETRAIN_MODEL_PATH="${MODEL_PATH}}/ckp.pt"
ROOT_DIR="/home/ldlqudgus756/cl_pathology/data"

DATASET_PATH="${ROOT_DIR}/${DATASETS}/#organ#/extract_patch512_512_w_patch_labels_mixed/${BACKBONE}_trunc"
WSI_PATH="${ROOT_DIR}/${DATASETS}/#organ#/WSIs"

CUDA_VISIBLE_DEVICES=0 python3 ./main/single_default/run.py \
    --project=${PROJECT_NAME} \
    --datasets=${DATASETS} \
    --organ=${ORGAN} \
    --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
    --dataset_root=${DATASET_PATH} \
    --model_path=${MODEL_PATH} \
    --cv_fold=5 \
    --model ${MODEL} \
    --pool=attn \
    --n_trans_layers=${NLAYER} \
    --da_act=tanh \
    --title=${TITLE} \
    --epeg_k=15 \
    --crmsa_k=1 \
    --all_shortcut \
    --seed=2021 \
    --lr 1e-4 \
    --rank ${RANK} \
    --wo_pretrain \
    --train_patch_epoch=0 \
    --num_epoch 100 \
    --cls_alpha 1 \
    --loss "ce" \
    --seg_coef 10.0 \
    --wsi_path ${WSI_PATH} \
    --ssl_config_file ./ssl_engine/ssl_config/histo_mil/supervised/supervised_paip_mil.yaml \
    --per_seg_vis_epoch 4 \
    --disable_vis_seg \
    --vis_sample_interval 32 \
    --noise_scale 0.005 \
    --input_dim 1024 \
    --wandb \
    --n_tasks 3 \
    --cl_lr 1e-4 \
    --fold_start 0 \
    --fold_end 5 \
    --st_task 0



