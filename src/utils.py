import pydicom
import nibabel as nib
import numpy as np
import os
import torch
import random
from losses import SoftDiceLoss, SoftDiceBCECombinedLoss
import torch.nn as nn
from models import SimpleUNet, UNet2D

# TODO: De identify patients patientname, patientid, patientdob, patientsex


def load_npz(file_path, dtype=np.int16):
    # Load segmentation data from .npz file.
    data = np.load(file_path)
    assert len(data.files) == 1, "The .npz file should contain only one array"
    tag = data.files[0]
    arr = data[tag].astype(dtype)
    return arr


def dicom_dir_to_3d_arr(dicom_dir, dtype=np.int16):
    # Load all DICOM files in the given directory and sort them by slice
    # position.
    files = [
        pydicom.dcmread(os.path.join(dicom_dir, f))
        for f in os.listdir(dicom_dir)
        if f.endswith(".dcm")
    ]
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Stack the pixel arrays to create a 3D array.
    arr = np.stack([f.pixel_array for f in files]).astype(dtype)

    return arr


def make_niftis(img_dir, seg_f, out_dir):
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


def set_device():
    # Check for CUDA, then MPS, and default to CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def seed_everything(seed):
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


def load_slices_from_dataset(img_dir, mask_dir, case_ids=None):
    # Load the 3D volume and mask for each case.
    # Expects the dataset to be in the format of:
    # Given in the cw.
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


def get_losses_dict():
    # Define the loss functions.
    loss_fns = {
        "dice": SoftDiceLoss,
        "bce": nn.BCEWithLogitsLoss,
        "dice_and_bce": SoftDiceBCECombinedLoss,
    }

    return loss_fns


def get_model_dict():
    # Define the model classes.
    model_dict = {
        "unet": UNet2D,
        "demo_unet": SimpleUNet,
    }

    return model_dict


def safe_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        print(f"[INFO] {dir_path} does not exit, making dir(s)")
        os.makedirs(dir_path)
    else:
        if os.listdir(dir_path):
            print(f"[ERROR] Files exist in {dir_path}... exiting")
            exit(1)
