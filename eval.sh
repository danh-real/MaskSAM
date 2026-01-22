#!/bin/bash

declare -a datasets=(
    "Dataset001_BrainTumour"
    "Dataset002_Heart"
    "Dataset003_Liver"
    "Dataset004_Hippocampus"
    "Dataset010_Colon"
    "Dataset201_Sarcoma"
)

for dataset in ${datasets[@]};
do
    python eval.py $dataset
done