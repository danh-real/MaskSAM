#!/bin/bash

export nnUNet_raw="/data/rlsam2/MaskSAM/DATASETS/nnUnet_raw"
export nnUNet_preprocessed="/data/rlsam2/MaskSAM/DATASETS/nnUnet_preprocessed"
export nnUNet_results="/data/rlsam2/MaskSAM/DATASETS/nnUnet_results"

# CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -f all -tr MaskSAM_AMOS -c 3d_fullres \
#     -i /data/rlsam2/MaskSAM/DATASETS/nnUnet_raw/Dataset002_Heart/imagesTs \
#     -o /data/rlsam2/MaskSAM/DATASETS/nnUnet_results/Dataset002_Heart/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
#     -d 002 \
#     -chk checkpoint_best.pth

# CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -f all -tr MaskSAM_AMOS -c 3d_fullres \
#     -i /data/rlsam2/MaskSAM/DATASETS/nnUnet_raw/Dataset003_Liver/imagesTs \
#     -o /data/rlsam2/MaskSAM/DATASETS/nnUnet_results/Dataset003_Liver/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
#     -d 003 \
#     -chk checkpoint_best.pth

CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -f all -tr MaskSAM_AMOS -c 3d_fullres \
    -i /data/rlsam2/MaskSAM/DATASETS/nnUnet_raw/Dataset004_Hippocampus/imagesTs \
    -o /data/rlsam2/MaskSAM/DATASETS/nnUnet_results/Dataset004_Hippocampus/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
    -d 004 \
    -chk checkpoint_best.pth

CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -f all -tr MaskSAM_AMOS -c 3d_fullres \
    -i /data/rlsam2/MaskSAM/DATASETS/nnUnet_raw/Dataset010_Colon/imagesTs \
    -o /data/rlsam2/MaskSAM/DATASETS/nnUnet_results/Dataset010_Colon/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
    -d 010 \
    -chk checkpoint_best.pth