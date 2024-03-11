import os
import argparse
import yaml
import wandb
from tqdm import tqdm
import torch
from torchvision import transforms
from torchmetrics.classification import BinaryAccuracy
import matplotlib.pyplot as plt
from accelerate import Accelerator
from datasets import ImageSegmentationDataset
from utils import (
    set_device,
    seed_everything,
    get_losses_dict,
    load_slices_from_dataset,
    get_model_dict,
    safe_create_dir,
)


def main(config):
    # Check if models_temp_dir exists
    model_save_dir = config["model_save_dir"]

    # Creates a dir if one does not exist. Will not overwrite an existing dir.
    safe_create_dir(model_save_dir)

    # If .git file exists, add git hash to config.
    if os.path.exists(".git"):
        config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    config_file = os.path.join(model_save_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Set random seed for reproducability.
    seed_everything(config["seed"])

    # Fetch device.
    device = set_device()

    if config["wandb_log"]:
        # Start a new wandb run to track this train job.
        wandb.init(
            project="medical_imaging_1",
            config=config,
        )

    print("[INFO] Device set to:", device)
    print("-" * 50)
    print("[INFO] Config options set.")
    for key, val in config.items():
        print(f"[INFO] {key}: {val}")
    print("-" * 50)

    # Load the the slices from the volume and mask of each case.
    train_slices = load_slices_from_dataset(
        config["img_dir"],
        config["mask_dir"],
        case_ids=config["case_ids"],
    )

    # If only foreground slices specified, remove entres with all zero masks.
    if config["only_foreground_slices"]:
        train_slices = [
            case_tuple for case_tuple in train_slices if case_tuple[2].any()
        ]

    if config["invert_masks"]:
        mask_dtype = train_slices[0][2].dtype
        # Casts the mask to a boolean array and performs a bitwise not operation.
        # to invert the array, then casts it back to the original dtype.
        train_slices = [
            (case_id, img, (~mask.astype(bool)).astype(mask_dtype))
            for case_id, img, mask in train_slices
        ]

    # Unpack case ids, images and masks from the train list.
    train_case_ids, train_images, train_masks = list(zip(*train_slices))

    if config["quick_test"]:
        # pick central slice since it normally contains a slice with a large
        # amount of the lungs.
        print("[INFO] Quick test mode enabled.")
        print("[INFO] Only using only one case for training and testing.")
        mid_train_idx = len(train_case_ids) // 2
        train_case_ids = [train_case_ids[mid_train_idx]]
        train_images = [train_images[mid_train_idx]]
        train_masks = [train_masks[mid_train_idx]]

    # Define train data transforms.
    # TODO: Define transforms in config file.
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Split the dataset into train and validation sets.
    train_dataset = ImageSegmentationDataset(
        train_images,
        train_masks,
        train_case_ids,
        image_transform=train_transforms,
        mask_transform=train_transforms,
    )

    # Define the dataloaders.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    # Define model, loss function, and optimiser.
    model_dict = get_model_dict()
    model = model_dict[config["model_str"]](in_channels=1, out_channels=1)

    if config["model_weights_path"]:
        model.load_state_dict(
            torch.load(config["model_weights_path"], map_location=torch.device("cpu"))
        )
        print(f"[INFO] Loaded model weights from {config['model_weights_path']}")

    loss_fn_dict = get_losses_dict()
    loss_fn = loss_fn_dict[config["loss"]]()

    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Define the mask labels for wandb visualisations.
    class_labels = (
        {0: "background", 1: "lung"}
        if config["invert_masks"]
        else {0: "lung", 1: "background"}
    )

    # Move model, optim and dataloader to accelerator.
    accelerator = Accelerator()
    model, optim, dataloader = accelerator.prepare(model, optim, train_loader)

    # Define additional performance metrics.
    # TODO: try putting metrics in accelerator.
    metric = BinaryAccuracy().to(device)

    # Train the model.
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}"
        )

        for i, (images, masks, _) in progress_bar:
            optim.zero_grad()
            preds = model(images)
            loss = loss_fn(preds.to(torch.float32), masks.to(torch.float32)).mean()
            loss.backward()
            optim.step()

            metric.update(preds, masks)
            accuracy = metric.compute()

            if config["wandb_log"]:
                wandb.log({"loss": loss.item()})
                wandb.log({"accuracy": accuracy})

            total_loss += loss.item()
            total_acc += accuracy
            num_batches += 1

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        summary_stats = (
            f"Epoch: {epoch}, Avg. Loss: {avg_loss:.4f}, Avg. Accuracy: {avg_acc:.4f}"
        )
        print(summary_stats)

        # Save the model weights.
        model_str = f"epoch_{epoch}_model.pt"
        model_path = os.path.join(model_save_dir, model_str)
        torch.save(model.state_dict(), model_path)

        # Visualise predictions vs ground truth
        # TODO: Find a good batch to visualise that has non zero masks.
        img = images[0].squeeze().cpu().detach().numpy()
        mask = masks[0].squeeze().cpu().detach().numpy()
        pred = (torch.sigmoid(preds[0]) > 0.5).squeeze().cpu().detach().numpy()

        if config["visualise"]:
            fig, ax = plt.subplots(ncols=3)
            ax[0].imshow(mask)
            ax[0].set_title("Ground Truth")
            ax[1].imshow(pred)
            ax[1].set_title("Prediction")
            ax[2].imshow(img, cmap="gray")
            ax[2].set_title("Image")
            plt.show()

        if config["wandb_log"]:
            wandb.log(
                {
                    "lung_ct_seg_test": wandb.Image(
                        img,
                        masks={
                            "predictions": {
                                "mask_data": pred,
                                "class_labels": class_labels,
                            },
                            "ground_truth": {
                                "mask_data": mask,
                                "class_labels": class_labels,
                            },
                        },
                    )
                }
            )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to training yaml config file")
    args = parser.parse_args()

    # load yaml config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
