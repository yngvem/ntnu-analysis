# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Generator, Iterable, List, Set, Tuple, Union

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

COMPRESSION_OPTS = {"compression": "lzf"}


def load_nii(nii_file: Union[Path, str]) -> np.array:
    """Return the contents of a nii file as a NumPy array.
    """
    return np.array(nib.load(nii_file).get_fdata())


def load_dwi(subject: int, downsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load the images and mask for a DWI scan folder.
    """
    dwi_image_names = [f"img_b{b}.nii" for b in range(7)]
    dwi_mask_names = ["Mask_an_transformed.nii", "Mask_shh_transformed.nii"]

    images = np.stack(
        [
            load_nii(subject / "DWI" / image_name).T[:, ::downsample, ::downsample]
            for image_name in dwi_image_names
        ],
        axis=-1,
    )

    masks = np.stack(
        [
            load_nii(subject / "DWI" / mask_name).T[:, ::downsample, ::downsample]
            for mask_name in dwi_mask_names
        ],
        axis=-1,
    )
    return images, masks


def load_t2w(subject: int, downsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load the images and mask for a T2W scan folder.
    """
    dwi_image_names = [f"img.nii"]
    dwi_mask_names = ["Mask_an.nii", "Mask_shh.nii"]

    images = np.stack(
        [
            load_nii(subject / "T2" / image_name).T[:, ::downsample, ::downsample]
            for image_name in dwi_image_names
        ],
        axis=-1,
    )

    masks = np.stack(
        [
            load_nii(subject / "T2" / mask_name).T[:, ::downsample, ::downsample]
            for mask_name in dwi_mask_names
        ],
        axis=-1,
    )
    return images, masks


def load_mix(subject: int, downsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load the images and mask for a fused DWI+T2W folder.
    """
    image_names = [subject / f"T2/img.nii"] + [
        subject / f"DWI_onT2/img_b{b}.nii" for b in range(7)
    ]
    t2w_mask_names = ["Mask_an.nii", "Mask_shh.nii"]

    images = np.stack(
        [
            load_nii(image_name).T[:, ::downsample, ::downsample]
            for image_name in image_names
        ],
        axis=-1,
    )

    masks = np.stack(
        [
            load_nii(subject / "T2" / mask_name).T[:, ::downsample, ::downsample]
            for mask_name in t2w_mask_names
        ],
        axis=-1,
    )
    return images, masks


def populate_initial_dataset(
    data: np.ndarray, h5: h5py.Group, dataset_name: str
) -> h5py.Dataset:
    """Initialise a dataset in the input HDF5 group.
    Used to store all images in a split.
    """
    shape = data.shape
    maxshape = (None, *shape[1:])
    dataset = h5.create_dataset(
        name=dataset_name,
        data=data,
        dtype=data.dtype,
        shape=shape,
        maxshape=maxshape,
        chunks=(1, *shape[1:]),
        **COMPRESSION_OPTS,
    )
    return dataset


def extend_dataset(dataset: h5py.Dataset, data: np.ndarray,) -> h5py.Dataset:
    """Ectend a dataset in the input HDF5 group. 
    Used to update the images in a split.
    """
    shape = dataset.shape
    newshape = (dataset.shape[0] + data.shape[0], *dataset.shape[1:])
    dataset.resize(newshape)
    dataset[shape[0] :] = data
    return dataset


def get_patient_id(patient: Path) -> int:
    """Return the patient ID from the path of this patient's data folder.
    """
    return int(patient.name.split("_")[1])


def generate_fold_group(
    group: h5py.Group, patient_folders: Iterable[Path], load_f: Callable
) -> None:
    """Generate dataset for a single fold.
    """
    input_dataset = None
    mask_dataset = None
    patient_ids = None
    for patient in tqdm(patient_folders):
        images, masks = load_f(patient)
        patient_id = get_patient_id(patient)
        patient_id = np.ones(images.shape[0]) * patient_id

        if input_dataset is None:
            input_dataset = populate_initial_dataset(images, group, "input")
            mask_dataset_an = populate_initial_dataset(
                masks[..., 0, np.newaxis], group, "target_an"
            )
            mask_dataset_shh = populate_initial_dataset(
                masks[..., 1, np.newaxis], group, "target_shh"
            )
            mask_dataset_union = populate_initial_dataset(
                masks.max(axis=-1, keepdims=True), group, "target_union"
            )
            mask_dataset_intersection = populate_initial_dataset(
                masks.min(axis=-1, keepdims=True), group, "target_intersection"
            )
            patient_ids = populate_initial_dataset(patient_id, group, "patient_ids")
        else:
            input_dataset = extend_dataset(input_dataset, images)
            mask_dataset_an = extend_dataset(mask_dataset_an, masks[..., 0, np.newaxis])
            mask_dataset_shh = extend_dataset(
                mask_dataset_shh, masks[..., 1, np.newaxis]
            )
            mask_dataset_union = extend_dataset(
                mask_dataset_union, masks.max(axis=-1, keepdims=True)
            )
            mask_dataset_intersection = extend_dataset(
                mask_dataset_intersection, masks.min(axis=-1, keepdims=True)
            )
            patient_ids = extend_dataset(patient_ids, patient_id)


def patient_iter(
    data_folder: Union[Path, str], id_list: List[int]
) -> Generator[Path, None, None]:
    """Iterate over the patient folders corresponding to the input id list.
    """
    data_folder = Path(data_folder)
    for id_ in id_list:
        yield data_folder / f"Oxytarget_{id_}_PRE"


def random_split(ids: Iterable[int], n_folds: int) -> List[Set[int]]:
    """Randomly split the data into n equally sized folds.
    
    If data isn't divisible by the number of folds, then the last fold
    will have more items than the rest.
    """
    ids = list(ids)
    n_per_fold = int(len(ids) / n_folds)

    random.shuffle(ids)
    folds = [
        set(ids[fold_num * n_per_fold : (fold_num + 1) * n_per_fold])
        for fold_num in range(n_folds)
    ]

    missed_ids = len(ids) - n_per_fold * n_folds
    if missed_ids > 0:
        folds[-1] |= set(ids[-missed_ids:])

    return folds


def generate_folds(
    splits: Dict[str, Set[int]], num_per_fold: int,
) -> Dict[str, List[Set[int]]]:
    """Generate training, testing and validation folds based on input.
    """
    folds = {}
    for split, ids in splits.items():
        num_folds = int(len(ids) / num_per_fold)
        folds[split] = random_split(ids, num_folds)

    return folds


def generate_hdf5_file(
    folds: Dict[str, List[Set[int]]],
    out_name: str,
    data_path: Path,
    load_f: Callable,
    overwrite=False,
) -> DefaultDict[str, List[str]]:
    """Generate a HDF5 file based on dataset splits.
    
    fold_names is a dictionary that maps the split names to a list of
    folds names in each split.
    """
    fold_names = defaultdict(list)

    out_file = data_path / out_name
    if not overwrite and out_file.is_file():
        raise RuntimeError("File exists")

    fold_num = 0
    with h5py.File(out_file, "w") as h5:
        for split in folds:  # split is usually train, test or val
            print(split)

            for fold in folds[split]:
                foldname = f"fold_{fold_num}"
                print(foldname)
                fold_num += 1

                # Update h5py
                group = h5.create_group(foldname)
                fold_names[split].append(foldname)

                fold = sorted(patient_iter(data_path, fold))
                generate_fold_group(group, fold, load_f)

    return fold_names


if __name__ == "__main__":
    splits = {
        "train": {
            27,
            29,
            32,
            40,
            41,
            45,
            47,
            48,
            49,
            52,
            55,
            64,
            67,
            68,
            69,
            73,
            75,
            77,
            79,
            80,
            85,
            87,
            89,
            94,
            95,
            96,
            115,
            116,
            118,
            120,
            121,
            127,
            138,
            145,
            146,
            150,
            153,
            155,
            163,
            165,
            171,
            172,
            173,
            174,
            175,
            177,
            184,
            185,
            186,
            187,
            191,
        },
        "val": {72, 74, 88, 124, 125, 128, 148, 156, 157, 164},
        "test": {
            31,
            43,
            44,
            46,
            56,
            57,
            65,
            83,
            90,
            103,
            131,
            133,
            134,
            143,
            144,
            160,
            166,
            176,
            181,
            189,
        },
    }

    folds = generate_folds(splits, 10)
    data_path = Path(r"W:\Data_Eline_Ås\Data_Eline_Ås")

    generate_hdf5_file(
        folds,
        out_name="t2w_compress_downsample8.h5",
        data_path=data_path,
        load_f=partial(load_t2w, downsample=8),
        overwrite=True,
    )
