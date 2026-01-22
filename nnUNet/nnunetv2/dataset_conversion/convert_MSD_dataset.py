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


def split_4d_nifti(filename, output_folder):
    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = os.path.basename(filename)
    if dim == 3:
        shutil.copy(filename, join(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, filename))
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4,4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(img_itk_new, join(output_folder, file_base[:-7] + "_%04.0d.nii.gz" % i))


def convert_msd_dataset(source_folder: str, overwrite_target_id: Optional[int] = None,
                        num_processes: int = default_num_processes) -> None:
    if source_folder.endswith('/') or source_folder.endswith('\\'):
        source_folder = source_folder[:-1]

    labelsTr = join(source_folder, 'labelsTr')
    imagesTs = join(source_folder, 'imagesTs')
    imagesTr = join(source_folder, 'imagesTr')
    assert isdir(labelsTr), f"labelsTr subfolder missing in source folder"
    assert isdir(imagesTs), f"imagesTs subfolder missing in source folder"
    assert isdir(imagesTr), f"imagesTr subfolder missing in source folder"
    dataset_json = join(source_folder, 'dataset.json')
    assert isfile(dataset_json), f"dataset.json missing in source_folder"

    # infer source dataset id and name
    task, dataset_name = os.path.basename(source_folder).split('_')
    task_id = int(task[4:])

    # check if target dataset id is taken
    target_id = task_id if overwrite_target_id is None else overwrite_target_id
    existing_datasets = find_candidate_datasets(target_id)
    assert len(existing_datasets) == 0, f"Target dataset id {target_id} is already taken, please consider changing " \
                                        f"it using overwrite_target_id. Conflicting dataset: {existing_datasets} (check nnUNet_results, nnUNet_preprocessed and nnUNet_raw!)"
    target_dataset_name = f"Dataset{target_id:03d}_{dataset_name}"
    target_folder = join(nnUNet_raw, target_dataset_name)
    target_imagesTr = join(target_folder, 'imagesTr')
    target_imagesTs = join(target_folder, 'imagesTs')
    target_labelsTr = join(target_folder, 'labelsTr')
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        # convert 4d train images
        source_images = [i for i in subfiles(imagesTr, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [join(imagesTr, i) for i in source_images]

        results.append(
            p.starmap_async(
                split_4d_nifti, zip(source_images, [target_imagesTr] * len(source_images))
            )
        )

        # convert 4d test images
        source_images = [i for i in subfiles(imagesTs, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [join(imagesTs, i) for i in source_images]

        results.append(
            p.starmap_async(
                split_4d_nifti, zip(source_images, [target_imagesTs] * len(source_images))
            )
        )

        # copy segmentations (train labels)
        source_images = [i for i in subfiles(labelsTr, suffix='.nii.gz', join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        for s in source_images:
            shutil.copy(join(labelsTr, s), join(target_labelsTr, s))

        # copy test labels if they exist (optional for MSD)
        labelsTs = join(source_folder, 'labelsTs')
        if isdir(labelsTs):
            target_labelsTs = join(target_folder, 'labelsTs')
            maybe_mkdir_p(target_labelsTs)
            source_images = [i for i in subfiles(labelsTs, suffix='.nii.gz', join=False) if
                             not i.startswith('.') and not i.startswith('_')]
            for s in source_images:
                shutil.copy(join(labelsTs, s), join(target_labelsTs, s))

        [i.get() for i in results]

    # Load and convert dataset.json
    dataset_json = load_json(dataset_json)
    
    # Handle labels - check if already in correct format
    if 'labels' in dataset_json:
        labels = dataset_json['labels']
        first_key = list(labels.keys())[0]
        
        # If labels are in format {"background": 0, ...}, convert to {"background": 0, ...}
        # If labels are in format {"0": "background", ...}, convert to {"background": 0, ...}
        if first_key.isdigit():
            # Old MSD format: {"0": "background", "1": "label1"} -> {"background": 0, "label1": 1}
            dataset_json['labels'] = {j: int(i) for i, j in labels.items()}
        else:
            # Already in new format: {"background": 0, "label1": 1} - keep as is
            pass
    dataset_json['file_ending'] = ".nii.gz"
    print(dataset_json)
    # Handle channel_names/modality field
    if "channel_names" not in dataset_json:
        if "modality" in dataset_json:
            # Old MSD format
            dataset_json["channel_names"] = dataset_json["modality"]
            del dataset_json["modality"]
        else:
            # No modality or channel_names found, create default
            print("Warning: No 'channel_names' or 'modality' field found in dataset.json. Creating default channel names.")
            # Infer number of channels from first image
            first_img = subfiles(target_imagesTr, suffix='.nii.gz', join=False)[0]
            # Count how many channel files exist for this case
            case_id = first_img.split('_')[0]
            num_channels = len([f for f in subfiles(target_imagesTr, join=False) if f.startswith(case_id)])
            dataset_json["channel_names"] = {str(i): f"channel_{i}" for i in range(num_channels)}
    
    # Remove old MSD-specific fields if they exist
    if "training" in dataset_json:
        del dataset_json["training"]
    if "test" in dataset_json:
        del dataset_json["test"]
    if "modality" in dataset_json:
        del dataset_json["modality"]
    
    save_json(dataset_json, join(nnUNet_raw, target_dataset_name, 'dataset.json'), sort_keys=False)
    
    print(f"✓ Successfully converted {os.path.basename(source_folder)} to {target_dataset_name}")
    print(f"  Location: {target_folder}")


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='Downloaded and extracted MSD dataset folder. CANNOT be nnUNetv1 dataset! Example: '
                             '/home/fabian/Downloads/Task05_Prostate')
    parser.add_argument('-overwrite_id', type=int, required=False, default=None,
                        help='Overwrite the dataset id. If not set we use the id of the MSD task (inferred from '
                             'folder name). Only use this if you already have an equivalently numbered dataset!')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'Number of processes used. Default: {default_num_processes}')
    args = parser.parse_args()
    convert_msd_dataset(args.i, args.overwrite_id, args.np)


if __name__ == '__main__':
    convert_msd_dataset('/home/fabian/Downloads/Task05_Prostate', overwrite_target_id=201)