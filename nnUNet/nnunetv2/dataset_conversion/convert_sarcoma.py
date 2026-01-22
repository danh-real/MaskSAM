import argparse
import multiprocessing
import shutil
from multiprocessing import Pool
from typing import Optional
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import find_candidate_datasets
from nnunetv2.configuration import default_num_processes
import numpy as np


def copy_3d_nifti(filename, output_folder):
    """Copy 3D NIfTI file and rename with _0000 suffix for nnUNet format"""
    file_base = os.path.basename(filename)
    # Remove .nii.gz extension and add _0000.nii.gz
    output_name = file_base[:-7] + "_0000.nii.gz"
    shutil.copy(filename, join(output_folder, output_name))


def convert_sarcoma_dataset(source_folder: str, overwrite_target_id: Optional[int] = None,
                            num_processes: int = default_num_processes) -> None:
    """
    Convert Sarcoma dataset to nnUNet format
    
    Args:
        source_folder: Path to Sarcoma dataset folder containing imagesTr, imagesTs, labelsTr, labelsTs
        overwrite_target_id: Target dataset ID for nnUNet (e.g., 201)
        num_processes: Number of parallel processes for conversion
    """
    if source_folder.endswith('/') or source_folder.endswith('\\'):
        source_folder = source_folder[:-1]

    # Check required folders
    labelsTr = join(source_folder, 'labelsTr')
    imagesTs = join(source_folder, 'imagesTs')
    imagesTr = join(source_folder, 'imagesTr')
    labelsTs = join(source_folder, 'labelsTs')
    
    assert isdir(labelsTr), f"labelsTr subfolder missing in source folder: {labelsTr}"
    assert isdir(imagesTs), f"imagesTs subfolder missing in source folder: {imagesTs}"
    assert isdir(imagesTr), f"imagesTr subfolder missing in source folder: {imagesTr}"
    
    dataset_json_path = join(source_folder, 'dataset.json')
    assert isfile(dataset_json_path), f"dataset.json missing in source_folder: {dataset_json_path}"

    # Get dataset name from folder
    dataset_name = os.path.basename(source_folder)
    
    # Check if target dataset id is taken
    if overwrite_target_id is None:
        raise ValueError("Please provide a target dataset ID using overwrite_target_id parameter")
    
    target_id = overwrite_target_id
    existing_datasets = find_candidate_datasets(target_id)
    assert len(existing_datasets) == 0, f"Target dataset id {target_id} is already taken, please consider changing " \
                                        f"it using overwrite_target_id. Conflicting dataset: {existing_datasets} (check nnUNet_results, nnUNet_preprocessed and nnUNet_raw!)"

    # Create target folder structure
    target_dataset_name = f"Dataset{target_id:03d}_{dataset_name}"
    target_folder = join(nnUNet_raw, target_dataset_name)
    target_imagesTr = join(target_folder, 'imagesTr')
    target_imagesTs = join(target_folder, 'imagesTs')
    target_labelsTr = join(target_folder, 'labelsTr')
    
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    print(f"Converting Sarcoma dataset to {target_dataset_name}...")
    
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        # Process train images (3D, single channel)
        print("Processing training images...")
        source_images = [i for i in subfiles(imagesTr, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images_full = [join(imagesTr, i) for i in source_images]
        
        results.append(
            p.starmap_async(
                copy_3d_nifti, zip(source_images_full, [target_imagesTr] * len(source_images_full))
            )
        )

        # Process test images (3D, single channel)
        print("Processing test images...")
        source_images = [i for i in subfiles(imagesTs, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images_full = [join(imagesTs, i) for i in source_images]
        
        results.append(
            p.starmap_async(
                copy_3d_nifti, zip(source_images_full, [target_imagesTs] * len(source_images_full))
            )
        )

        # Wait for image processing to complete
        [i.get() for i in results]

    # Copy segmentation labels (train)
    print("Copying training labels...")
    source_labels = [i for i in subfiles(labelsTr, suffix='.nii.gz', join=False) if
                     not i.startswith('.') and not i.startswith('_')]
    for s in source_labels:
        shutil.copy(join(labelsTr, s), join(target_labelsTr, s))

    # Copy test labels if they exist
    if isdir(labelsTs):
        print("Copying test labels...")
        target_labelsTs = join(target_folder, 'labelsTs')
        maybe_mkdir_p(target_labelsTs)
        source_labels = [i for i in subfiles(labelsTs, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        for s in source_labels:
            shutil.copy(join(labelsTs, s), join(target_labelsTs, s))

    # Load and process dataset.json
    print("Processing dataset.json...")
    dataset_json = load_json(dataset_json_path)
    
    # Ensure labels are in correct nnUNet v2 format: {"label_name": int_value}
    if 'labels' in dataset_json:
        labels = dataset_json['labels']
        first_key = list(labels.keys())[0]
        
        # If labels are in old format {"0": "background", "1": "tumor"}, convert to new format
        if first_key.isdigit():
            dataset_json['labels'] = {j: int(i) for i, j in labels.items()}
    
    # Ensure channel_names exist
    if "channel_names" not in dataset_json:
        # Sarcoma dataset is single-channel MRI
        dataset_json["channel_names"] = {"0": "MRI"}
        print("Warning: channel_names not found in dataset.json. Set to single MRI channel.")
    
    # Set file ending
    dataset_json['file_ending'] = ".nii.gz"
    
    # Remove old/unnecessary fields
    for field in ["training", "test", "modality", "tensorImageSize", "reference", "licence", "release"]:
        if field in dataset_json:
            del dataset_json[field]
    
    # Keep important metadata fields
    if "description" not in dataset_json:
        dataset_json["description"] = "Soft-Tissue-Sarcoma dataset from TCIA"
    
    if "numTraining" not in dataset_json:
        dataset_json["numTraining"] = len(source_labels)
    
    # Save the converted dataset.json
    save_json(dataset_json, join(target_folder, 'dataset.json'), sort_keys=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully converted Sarcoma dataset to {target_dataset_name}")
    print(f"  Location: {target_folder}")
    print(f"  Training images: {len([f for f in subfiles(target_imagesTr) if f.endswith('.nii.gz')])}")
    print(f"  Test images: {len([f for f in subfiles(target_imagesTs) if f.endswith('.nii.gz')])}")
    print(f"  Training labels: {len([f for f in subfiles(target_labelsTr) if f.endswith('.nii.gz')])}")
    if isdir(join(target_folder, 'labelsTs')):
        print(f"  Test labels: {len([f for f in subfiles(join(target_folder, 'labelsTs')) if f.endswith('.nii.gz')])}")
    print(f"{'='*80}\n")


def entry_point():
    parser = argparse.ArgumentParser(
        description='Convert Sarcoma dataset to nnUNet format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to Sarcoma dataset folder containing imagesTr, imagesTs, labelsTr, labelsTs. '
                             'Example: /path/to/Sarcoma')
    parser.add_argument('-d', '--dataset_id', type=int, required=True,
                        help='Target dataset ID for nnUNet (e.g., 201). Must be unique!')
    parser.add_argument('-np', '--num_processes', type=int, required=False, default=default_num_processes,
                        help=f'Number of processes for parallel conversion.')
    args = parser.parse_args()
    
    convert_sarcoma_dataset(args.input, args.dataset_id, args.num_processes)


if __name__ == '__main__':
    # Example usage - update paths as needed
    convert_sarcoma_dataset(
        source_folder='/data/rlsam2/MaskSAM/DATASETS/nnUnet_raw/Sarcoma',
        overwrite_target_id=201,
        num_processes=8
    )