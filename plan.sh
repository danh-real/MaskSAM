#!/bin/bash

export nnUNet_raw="/data/rlsam2/MaskSAM/DATASETS/nnUnet_raw"
export nnUNet_preprocessed="/data/rlsam2/MaskSAM/DATASETS/nnUnet_preprocessed"
export nnUNet_results="/data/rlsam2/MaskSAM/DATASETS/nnUnet_results"

# nnUNetv2_plan_and_preprocess -d 001 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 002 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 003 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 004 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 005 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 006 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 007 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 008 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 009 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 010 -c 3d_fullres --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 201 -c 3d_fullres --verify_dataset_integrity