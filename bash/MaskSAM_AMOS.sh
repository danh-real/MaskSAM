#!/bin/bash

export nnUNet_raw="/data/code/MaskSAM/DATASETS/nnUnet_raw"
export nnUNet_preprocessed="/data/code/MaskSAM/DATASETS/nnUnet_preprocessed"
export nnUNet_results="/data/code/MaskSAM/DATASETS/nnUnet_results"

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 201 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 002 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 004 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 010 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 005 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c