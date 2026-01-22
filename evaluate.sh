#!/bin/bash

export nnUNet_raw="/data/code/MaskSAM/DATASETS/nnUnet_raw"
export nnUNet_preprocessed="/data/code/MaskSAM/DATASETS/nnUnet_preprocessed"
export nnUNet_results="/data/code/MaskSAM/DATASETS/nnUnet_results"

CUDA_VISIBLE_DEVICES=1 nnUNetv2_evaluate_simple \
    /data/code/MaskSAM/DATASETS/nnUnet_raw/Dataset002_Heart/labelsTs \
    /data/code/MaskSAM/DATASETS/nnUnet_results/Dataset002_Heart/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
    -l 1

CUDA_VISIBLE_DEVICES=1 nnUNetv2_evaluate_simple \
    /data/code/MaskSAM/DATASETS/nnUnet_raw/Dataset003_Liver/labelsTs \
    /data/code/MaskSAM/DATASETS/nnUnet_results/Dataset003_Liver/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
    -l 1 2

CUDA_VISIBLE_DEVICES=0 nnUNetv2_evaluate_simple \
    /data/code/MaskSAM/DATASETS/nnUnet_raw/Dataset004_Hippocampus/labelsTs \
    /data/code/MaskSAM/DATASETS/nnUnet_results/Dataset004_Hippocampus/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
    -l 1 2

CUDA_VISIBLE_DEVICES=0 nnUNetv2_evaluate_simple \
    /data/code/MaskSAM/DATASETS/nnUnet_raw/Dataset010_Colon/labelsTs \
    /data/code/MaskSAM/DATASETS/nnUnet_results/Dataset010_Colon/MaskSAM_AMOS__nnUNetPlans__3d_fullres/predict \
    -l 1