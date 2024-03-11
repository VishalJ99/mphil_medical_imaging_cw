from utils import dicom_dir_to_3d_arr
from utils import load_npz
import nibabel as nib
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

# TODO:
# Add documentation re assumptions on file paths and assumed structure
# of output_dir and seg_dir.


def main(case_slice_id, img_dir, mask_dir, output_dir):
    # Get the case id and slice index from the case_slice_id.
    case_slice_id_split = case_slice_id.split("_")
    case_id, slice_idx = "_".join(case_slice_id_split[:-1]), case_slice_id_split[-1]

    # Construct the paths
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    seg_dir = os.path.join(output_dir, "segmentations")
    case_dir = f"{img_dir}/{case_id}"
    mask_path = f"{mask_dir}/{case_id}_seg.npz"
    seg_path = f"{seg_dir}/{case_id}_{slice_idx}_pred.nii.gz"

    # Load the arrays.
    case_arr = dicom_dir_to_3d_arr(case_dir)
    mask_arr = load_npz(mask_path)
    seg_slice = nib.load(seg_path).get_fdata()

    case_slice = case_arr[int(slice_idx)]
    mask_slice = mask_arr[int(slice_idx)]

    # Load the metrics csv and filter the row for the given case slice.
    df = pd.read_csv(metrics_csv_path)
    filtered_df = df[df["slice"] == case_slice_id]

    # plot the img, mask and seg.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(case_slice, cmap="gray")
    ax[0].set_title("Case")
    ax[1].imshow(mask_slice, cmap="gray")
    ax[1].set_title("Mask")
    ax[2].imshow(seg_slice, cmap="gray")
    ax[2].set_title("Segmentation")

    # Show the metrics for the given case slice.
    df_info = ""
    if not filtered_df.empty:
        # Formatting dice and accuracy to 4 decimal places
        for col in filtered_df.columns[1:]:
            print(col)
            value = filtered_df.iloc[0][col]
            df_info += f" {col} : {value:.4f}\n"

    fig.text(0.5, 0, df_info, ha="center")

    fig.suptitle(f"Case: {case_id}, Slice: {slice_idx}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case_slice_id", type=str, help="The case slice id to visualise."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory containing the metrics.csv file\
        and the segmentations directory.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        help="Path to the directory containing the images.",
        default="Dataset/Images",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        help="Path to the directory containing the masks.",
        default="Dataset/Segmentations",
    )
    args = parser.parse_args()

    main(args.case_slice_id, args.img_dir, args.mask_dir, args.output_dir)
