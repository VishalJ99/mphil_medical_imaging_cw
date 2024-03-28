import pydicom
import nibabel as nib
import numpy as np
import os
import torch
import random
from losses import SoftDiceLoss, SoftDiceBCECombinedLoss
import torch.nn as nn
from models import SimpleUNet, UNet2D
from typing import List, Tuple

# TODO: De identify patients patientname, patientid, patientdob, patientsex


def load_npz(file_path, dtype=np.int16) -> np.ndarray:
    """
    Load a .npz file and return the array contained within it.
    Assumes that the .npz file contains only one array.

    Parameters
    ----------
    file_path : str
        The path to the .npz file.

    dtype : numpy.dtype
        The data type to cast the array to.

    Returns
    -------
    numpy.ndarray
        The array contained within the .npz file.
    """
    data = np.load(file_path)
    assert len(data.files) == 1, "The .npz file should contain only one array"
    tag = data.files[0]
    arr = data[tag].astype(dtype)
    return arr


def dicom_dir_to_3d_arr(dicom_dir: str, dtype=np.int16) -> np.ndarray:
    """
    Loads all the DICOM files in a directory, sorts them by the z coordinate as
    specified by the ImagePositionPatient tag and stacks them to create a 3D array.
    Converts the units to Hounsfield units.

    Parameters
    ----------
    dicom_dir : str
        The path to the directory containing the DICOM files.

    dtype : numpy.dtype
        The data type to cast the array to.

    Returns
    -------
    numpy.ndarray
        A 3D array containing the pixel arrays of the DICOM files.

    """
    files = [
        pydicom.dcmread(os.path.join(dicom_dir, f))
        for f in os.listdir(dicom_dir)
        if f.endswith(".dcm")
    ]
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Convert pixel arrays to Hounsfield units and stack them to create a 3D array.
    arr = np.stack([
        file.pixel_array * file.RescaleSlope + file.RescaleIntercept
        for file in files
    ]).astype(dtype)

    return arr


def make_niftis(img_dir: str, seg_f: str, out_dir: str) -> None:
    """
    Creates a NIfTI image for the 3D volume in the img_dir and the segmentation
    mask in the seg_f. Saves the NIfTI images in the out_dir. Creates an identity
    affine matrix for the NIfTI images.

    Parameters
    ----------
    img_dir : str
        The path to the directory containing the DICOM files.

    seg_f : str
        The path to the .npz file containing the segmentation mask.

    out_dir : str
        The path to the directory to save the NIfTI images.

    Returns
    -------
    None
    """
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    img_arr = dicom_dir_to_3d_arr(img_dir).astype(np.int16)
    seg_arr = load_npz(seg_f).astype(np.int16)

    # Create NIfTI images.
    img_nii = nib.Nifti1Image(img_arr, np.eye(4))
    seg_nii = nib.Nifti1Image(seg_arr, np.eye(4))

    # save the nifti images.
    img_nii_f = os.path.join(out_dir, "img.nii.gz")
    seg_nii_f = os.path.join(out_dir, "seg.nii.gz")
    nib.save(img_nii, img_nii_f)
    nib.save(seg_nii, seg_nii_f)


def seed_everything(seed: int) -> None:
    """
    Sets the random seed for reproducability.
    Sets seeds for numpy, torch and random.

    Parameters
    ----------
    seed : int
        The seed to set for the random number generators.

    Returns
    -------
    None
    """
    # Set `PYTHONHASHSEED` environment variable to the seed.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seed for all packages with built-in pseudo-raiAndom generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using CUDA, set also the below for determinism.
    if torch.cuda.is_available():
        # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

        # Ensures that the CUDA convolution uses deterministic algorithms.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_slices_from_dataset(
    img_dir: int, mask_dir: int, case_ids: List[str] = None
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Load a list of all 2D slices in  the 3D volume and mask for each case in the image and mask directories.
    Also
    Expects the dataset to be in the format of:
    
    img_dir:
        - Case_1
            - 1.dcm
            ...
        ...
    
    mask_dir:
        - Case_1_seg.npz
        ...
    

    Parameters
    ----------
    img_dir : str
        The path to the directory containing case directories of DICOM files.
    
    mask_dir : str
        The path to the directory containing the segmentation masks.
    
    case_ids : List[str]
        A list of case ids to load. If None, all cases in the img_dir will be loaded.
        These are the names of the case directories in the img_dir.
    
    Returns
    -------
    List[Tuple[str, np.ndarray, np.ndarray]]
        A list of tuples containing the case id, 2D image slice and 2D mask slice.
    
    """
    slices_list = []
    for case in case_ids:
        case_fpath = os.path.join(img_dir, case)
        mask_fpath = os.path.join(mask_dir, case + "_seg.npz")

        case_arr = dicom_dir_to_3d_arr(case_fpath, np.float32)
        mask_arr = load_npz(mask_fpath)

        # Create a tuple for each slice in the 3D volume.
        for slice_idx in range(case_arr.shape[0]):
            case_tuple = (
                f"{case}_{slice_idx}",
                case_arr[slice_idx],
                mask_arr[slice_idx],
            )
            slices_list.append(case_tuple)
    return slices_list


def get_losses_dict() -> dict:
    """
    Defines dictionary used by train.py to get the loss functions.
    Allows for easy addition of new loss functions which can then be
    passed as a string in the config file.

    Returns
    -------
    dict
        A dictionary containing the loss functions.
    """
    # Define the loss functions.
    loss_fns = {
        "dice": SoftDiceLoss,
        "bce": nn.BCEWithLogitsLoss,
        "dice_and_bce": SoftDiceBCECombinedLoss,
    }

    return loss_fns


def get_model_dict() -> dict:
    """
    Defines dictionary used by train.py to get the models.
    Allows for easy addition of new models which can then be
    passed as a string in the config file.
    """
    # Define the model classes.
    model_dict = {
        "unet": UNet2D,
        "demo_unet": SimpleUNet,
    }

    return model_dict


def safe_create_dir(dir_path: str) -> None:
    """
    Creates a directory if it does not exist.
    If the directory exists, checks if it is empty.
    If the directory is not empty, prints an error message and exits.
    
    Parameters
    ----------
    dir_path : str
        The path to the directory to create.
    
    Returns
    -------
    None
    """
    if not os.path.isdir(dir_path):
        print(f"[INFO] {dir_path} does not exit, making dir(s)")
        os.makedirs(dir_path)
    else:
        if os.listdir(dir_path):
            print(f"[ERROR] Files exist in {dir_path}... exiting")
            exit(1)
