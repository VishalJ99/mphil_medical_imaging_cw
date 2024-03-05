import pydicom
import nibabel as nib
import numpy as np
import os

# TODO: De identify patients patientname, patientid, patientdob, patientsex


def load_npz(file_path):
    # Load segmentation data from .npz file
    data = np.load(file_path)
    assert len(data.files) == 1, "The .npz file should contain only one array"
    tag = data.files[0]
    arr = data[tag]
    return arr


def dicom_dir_to_3d_arr(dicom_dir):
    # Load all DICOM files in the given directory and sort them by slice
    # position
    files = [
        pydicom.dcmread(os.path.join(dicom_dir, f))
        for f in os.listdir(dicom_dir)
        if f.endswith(".dcm")
    ]
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Stack the pixel arrays to create a 3D array
    arr = np.stack([f.pixel_array for f in files])

    return arr


def make_niftis(img_dir, seg_f, out_dir):
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    img_arr = dicom_dir_to_3d_arr(img_dir).astype(np.int16)
    seg_arr = load_npz(seg_f).astype(np.int16)

    # Create NIfTI images
    img_nii = nib.Nifti1Image(img_arr, np.eye(4))
    seg_nii = nib.Nifti1Image(seg_arr, np.eye(4))

    # save the nifti images
    img_nii_f = os.path.join(out_dir, "img.nii.gz")
    seg_nii_f = os.path.join(out_dir, "seg.nii.gz")
    nib.save(img_nii, img_nii_f)
    nib.save(seg_nii, seg_nii_f)
