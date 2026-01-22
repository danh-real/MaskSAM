#!/bin/bash

export nnUNet_raw="/data/rlsam2/MaskSAM/DATASETS/nnUnet_raw"
export nnUNet_preprocessed="/data/rlsam2/MaskSAM/DATASETS/nnUnet_preprocessed"
export nnUNet_results="/data/rlsam2/MaskSAM/DATASETS/nnUnet_results"

nnUNetv2_train 009 3d_fullres all -tr MaskSAM_AMOS -p nnUNetPlans -num_gpus 1 --c