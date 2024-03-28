import os
import argparse
from pydicom import dcmread


def main(dicom_dir):
    # Walk through dicom dir and de idenfity all files.
    for case_id in os.listdir(dicom_dir):
        subdir_path = os.path.join(dicom_dir, case_id) 
        if not os.path.isdir(subdir_path):
            continue
        for file_name in os.listdir(subdir_path):
            if file_name.endswith(".dcm"):
                dicom_file = os.path.join(subdir_path, file_name)
                metadata = dcmread(dicom_file)
                # Modify the tags that contain patient information
                metadata['PatientID'].value = case_id
                metadata['PatientName'].value = case_id
                metadata['PatientBirthDate'].value = ''
                # PatientBirthTime is optional, it might not be present 
                try:
                    del metadata['PatientBirthTime']
                except:
                    pass
                # Don't forget to save the changes
                metadata.save_as(dicom_file)
    print('[INFO] De-identification complete.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dicom_dir",
        type=str,
        help="Path to directory of case dirs containing DICOM files.",
    )
    args = parser.parse_args()

    main(args.dicom_dir)