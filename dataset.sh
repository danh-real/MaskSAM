#!/bin/bash

export nnUNet_raw="/data/code/MaskSAM/DATASETS/nnUnet_raw"
export nnUNet_preprocessed="/data/code/MaskSAM/DATASETS/nnUnet_preprocessed"
export nnUNet_results="/data/code/MaskSAM/DATASETS/nnUnet_results"

ROOT=/data/code/MaskSAM/DATASETS/nnUnet_raw

# nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task01_BrainTumour -np 8
# nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task02_Heart -np 8
# nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task03_Liver -np 8
# nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task04_Hippocampus -np 8
nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task05_Prostate -np 8
nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task06_Lung -np 8
nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task07_Pancreas -np 8
nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task08_HepaticVessel -np 8
nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task09_Spleen -np 8
# nnUNetv2_convert_MSD_dataset -i ${ROOT}/Task10_Colon -np 8


# python nnUNet/nnunetv2/dataset_conversion/convert_sarcoma.py