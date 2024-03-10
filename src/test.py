import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torchvision import transforms
from torchmetrics.classification import BinaryAccuracy
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from datasets import ImageSegmentationDataset
from utils import (
    set_device,
    seed_everything,
    load_slices_from_dataset,
    get_model_dict,
    safe_create_dir,
)
from losses import SoftDiceLoss
import nibabel as nib
import numpy as np
import pandas as pd


def main(config):
    # Create output directory if needed.
    output_dir = config["output_dir"]
    safe_create_dir(output_dir)
    if config["save_segmentations"]:
        output_seg_dir = os.path.join(output_dir, "segmentations")
        safe_create_dir(output_seg_dir)

    # Save config yaml to output_dir with git commit hash.
    config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()
    config_file = os.path.join(output_dir, "test_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Set random seed for reproducability.
    seed_everything(config["seed"])

    # Fetch device.
    device = set_device()

    # Load model.
    model_dict = get_model_dict()
    model = model_dict[config["model_str"]](in_channels=1, out_channels=1)
    model.load_state_dict(
        torch.load(config["model_weights_path"], map_location=torch.device("cpu"))
    )

    print("[INFO] Config options set.")
    for key, val in config.items():
        print(f"[INFO] {key}: {val}")

    # Load the the slices from the volume and mask of each case.
    all_slices_list = load_slices_from_dataset(
        config["img_dir"],
        config["mask_dir"],
    )

    # Split 2D slice data into train and test sets.
    train, test = train_test_split(all_slices_list, test_size=config["test_split"])

    if config["eval_train_split"]:
        # Use the train set for evaluation.
        test = train

    if config["only_foreground_slices"]:
        # Remove slices with all zero masks.
        test = [case_tuple for case_tuple in test if case_tuple[2].any()]

    if config["invert_masks"]:
        mask_dtype = train[0][2].dtype
        # Casts the mask to a boolean array and performs a bitwise not operation.
        # to invert the array, then casts it back to the original dtype.
        test = [
            (case_id, img, (~mask.astype(bool)).astype(mask_dtype))
            for case_id, img, mask in test
        ]

    # Unpack case ids, images and masks from the train list.
    test_case_ids, test_images, test_masks = list(zip(*test))

    if config["quick_test"]:
        # pick central slice since it normally contains a slice with a large
        # amount of the lungs.
        print("[INFO] Quick test mode enabled.")
        print("[INFO] Only using only one case for training and testing.")
        mid_test_idx = len(test_case_ids) // 2
        test_case_ids = [test_case_ids[mid_test_idx]]
        test_images = [test_images[mid_test_idx]]
        test_masks = [test_masks[mid_test_idx]]

    # Define train data transforms.
    # TODO: Define transforms in config file.
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Initialise the dataset.
    test_dataset = ImageSegmentationDataset(
        test_images,
        test_masks,
        test_case_ids,
        image_transform=test_transforms,
        mask_transform=test_transforms,
    )

    # Initialise the dataloaders.
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=True)

    # Move model and dataloader to accelerator.
    accelerator = Accelerator()
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    # Define additional performance metrics.
    metric = BinaryAccuracy().to(device)
    calc_dice_loss = SoftDiceLoss()

    metrics_dict = {}

    # Test the model.
    with torch.no_grad():
        metrics_dict = {}
        total_dice = 0.0
        total_accuracy = 0.0
        valid_dice_count = 0  # Count non-NaN dice scores
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (images, masks, case) in pbar:
            case_id = case[0]

            # Forward pass.
            preds = model(images)

            # Calculate metrics.
            metric.update(preds, masks)
            accuracy = metric.compute().item()

            # Set dice to nan if mask is all zeros.
            if masks.sum() == 0:
                dice = torch.nan
            else:
                dice_loss = calc_dice_loss(preds, masks)
                dice = (1 - dice_loss).item()
                # Update totals only if dice is not NaN
                total_dice += dice
                valid_dice_count += 1  # Increment valid dice count

            # Add to metrics dict.
            metrics_dict[case_id] = {"dice": dice, "accuracy": accuracy}

            if config["save_segmentations"]:
                pred = (
                    (torch.sigmoid(preds[0]) > 0.5)
                    .squeeze()
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.uint8)
                )
                nib.save(
                    nib.Nifti1Image(pred, np.eye(4)),
                    os.path.join(output_seg_dir, f"{case_id}_pred.nii.gz"),
                )
            # Update accuracy metrics.
            total_accuracy += accuracy
            dice_str = "NaN" if torch.isnan(torch.tensor(dice)) else f"{dice:.4f}"
            pbar.set_description(
                f"Case: {case[0]}, Dice: {dice_str}, Accuracy: {accuracy:.4f}"
            )

        # Log average metrics, avoiding division by zero
        avg_dice = total_dice / valid_dice_count if valid_dice_count > 0 else torch.nan
        avg_accuracy = total_accuracy / len(test_loader)
        avg_dice_str = "NaN" if np.isnan(avg_dice) else f"{avg_dice:.4f}"
        print(
            f"\nAverage Dice Score: {avg_dice_str},\
              Average Accuracy: {avg_accuracy:.4f}"
        )

    # Save results to a csv file.
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.index.name = "slice"  # Set the name of the index column to 'slice_id'
    df.reset_index(inplace=True)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to training yaml config file")
    args = parser.parse_args()

    # load yaml config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
